/**
 * fullscreen.js — SDG DISC Dashboard
 *
 * Watches for Plotly graphs added to the DOM by Dash/React callbacks
 * and injects a small expand button onto each graph card.
 * Clicking it uses the native browser Fullscreen API so the card
 * fills the screen, falling back to filling the browser window where
 * fullscreen is unavailable. The chart is resized to the new space and
 * restored to its original size on exit.
 */
(function () {
    'use strict';

    // Inline SVG icons: the old ⛶ glyph is missing from many system fonts.
    var ICON_EXPAND = '<svg viewBox="0 0 24 24" width="15" height="15" ' +
        'fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 ' +
        '2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3' +
        'M16 21h3a2 2 0 0 0 2-2v-3"/></svg>';
    var ICON_CLOSE  = '<svg viewBox="0 0 24 24" width="15" height="15" ' +
        'fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';

    /**
     * Walk up the DOM from a .js-plotly-plot element to find the
     * nearest ancestor card — identified by inline border-radius,
     * which is how all Python-generated graph cards are styled.
     */
    function findCard(plotEl) {
        var node  = plotEl.parentElement;
        var steps = 0;
        while (node && steps < 10) {
            var inlineStyle = node.getAttribute('style') || '';
            if (inlineStyle.indexOf('border-radius') !== -1) {
                return node;
            }
            node = node.parentElement;
            steps++;
        }
        return null;
    }

    /** Inject the expand button into a card (idempotent). */
    function addExpandBtn(card) {
        if (!card) return;
        if (card.dataset.fsReady) return;   // already processed
        card.dataset.fsReady = '1';

        // Ensure the card is positioned so the button can be absolute
        var pos = window.getComputedStyle(card).position;
        if (pos === 'static') card.style.position = 'relative';

        var btn = document.createElement('button');
        btn.className   = 'graph-fs-btn';
        btn.title       = 'Full screen';
        btn.setAttribute('aria-label', 'Full screen');
        btn.innerHTML   = ICON_EXPAND;

        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            if (!document.fullscreenElement && !expanded()) holdPlace(card);
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else if (expanded()) {
                setExpanded(null);
            } else if (card.requestFullscreen) {
                // Refused in embedded views and some policies: fall back to
                // filling the browser window.
                card.requestFullscreen().catch(function () {
                    setExpanded(card);
                });
            } else if (card.webkitRequestFullscreen) {   // desktop Safari
                card.webkitRequestFullscreen();
            } else {
                setExpanded(card);                       // iPhone Safari
            }
        });

        card.appendChild(btn);
    }

    /** Scan a subtree for any Plotly plots and wire up their cards. */
    function scan(root) {
        if (!root || root.nodeType !== 1) return;
        if (root.classList && root.classList.contains('js-plotly-plot')) {
            addExpandBtn(findCard(root));
        }
        var plots = root.querySelectorAll ? root.querySelectorAll('.js-plotly-plot') : [];
        plots.forEach(function (plot) { addExpandBtn(findCard(plot)); });
    }

    /**
     * Window-filling fallback for when the Fullscreen API is unavailable or
     * refused (iPhone Safari, embedded previews): the card is pinned over the
     * page with CSS (.graph-expanded in style.css).
     */
    function expanded() { return document.querySelector('.graph-expanded'); }

    /**
     * While a card is enlarged it leaves the page flow, so the page would
     * shorten and the reader would come back to a different scroll position.
     * Keep the card's space and remember where the reader was.
     */
    var held = null;
    function holdPlace(card) {
        var box = card.parentElement;
        held = { box: box, minHeight: box.style.minHeight, y: window.scrollY };
        box.style.minHeight = card.offsetHeight + 'px';
    }
    function releasePlace() {
        if (!held) return;
        held.box.style.minHeight = held.minHeight;
        window.scrollTo(0, held.y);
        held = null;
    }

    /**
     * position:fixed is relative to the nearest ancestor with a transform,
     * filter or running animation — e.g. the tab's .tab-fade-in entrance
     * leaves an identity transform behind — so the card would only fill that
     * ancestor. Neutralize those ancestors while expanded, then restore.
     */
    var neutralized = [];
    function freeAncestors(card) {
        // Up to and including <body>/<html>: the page styles give <body> an
        // identity transform, which alone is enough to trap the card.
        for (var n = card.parentElement; n; n = n.parentElement) {
            var cs = window.getComputedStyle(n);
            // An element with a transform animation still traps fixed
            // children even once finished. These are one-shot entrance
            // animations (page and tab fade-ins) whose end state is the
            // element's normal look, so switching them off is invisible;
            // they are deliberately not restored, as re-adding the animation
            // would replay the entrance.
            if (cs.animationName !== 'none') n.style.animation = 'none';
            if (cs.transform !== 'none' || cs.filter !== 'none' ||
                    cs.perspective !== 'none') {
                // !important beats a finished animation's fill-mode value
                // without restarting the animation (which would replay the
                // page entrance on close).
                var saved = {};
                ['transform', 'filter', 'perspective'].forEach(function (p) {
                    saved[p] = [n.style.getPropertyValue(p),
                                n.style.getPropertyPriority(p)];
                    n.style.setProperty(p, 'none', 'important');
                });
                neutralized.push([n, saved]);
            }
        }
    }
    function restoreAncestors() {
        neutralized.forEach(function (entry) {
            var n = entry[0], saved = entry[1];
            Object.keys(saved).forEach(function (p) {
                if (saved[p][0]) n.style.setProperty(p, saved[p][0], saved[p][1]);
                else n.style.removeProperty(p);
            });
        });
        neutralized = [];
    }

    function setExpanded(card) {
        var current = expanded();
        if (current) current.classList.remove('graph-expanded');
        restoreAncestors();
        document.body.classList.toggle('graph-expanded-open', !!card);
        if (card) {
            freeAncestors(card);
            card.classList.add('graph-expanded');
        }
        onFullscreenChange();
    }

    /**
     * Figures carry a fixed layout height (e.g. the radar is 700px), so a
     * plain resize leaves them small in the middle of the screen. On entry,
     * size each plot in the fullscreen card to the screen; on exit, put back
     * the size the figure was built with.
     */
    function fitPlots() {
        if (!window.Plotly) return;
        var fs = document.fullscreenElement || document.webkitFullscreenElement ||
                 expanded();
        document.querySelectorAll('.js-plotly-plot').forEach(function (el) {
            try {
                if (fs && fs.contains(el)) {
                    if (el.dataset.fsOrigHeight === undefined) {
                        // Figures without an explicit height are sized by
                        // their container, which is enlarged while
                        // expanded; remember the rendered height instead.
                        el.dataset.fsOrigHeight = el.layout.height ||
                            Math.round(el.getBoundingClientRect().height);
                        el.dataset.fsOrigWidth  = el.layout.width  || '';
                    }
                    var plots = fs.querySelectorAll('.js-plotly-plot').length;
                    Plotly.relayout(el, {
                        height: Math.floor((fs.clientHeight - 56) / plots),
                        width:  fs.clientWidth - 48,
                    });
                } else if (el.dataset.fsOrigHeight !== undefined) {
                    var h = el.dataset.fsOrigHeight, w = el.dataset.fsOrigWidth;
                    delete el.dataset.fsOrigHeight;
                    delete el.dataset.fsOrigWidth;
                    Plotly.relayout(el, {
                        height: h ? Number(h) : null,
                        width:  w ? Number(w) : null,
                        autosize: true,
                    });
                } else {
                    Plotly.Plots.resize(el);
                }
            } catch (e) {}
        });
    }

    /** Update button icons when fullscreen state changes. */
    function onFullscreenChange() {
        var inFS = !!(document.fullscreenElement || expanded());
        if (!inFS) setTimeout(releasePlace, 200);   // after the plot resizes
        document.querySelectorAll('.graph-fs-btn').forEach(function (btn) {
            btn.innerHTML = inFS ? ICON_CLOSE : ICON_EXPAND;
            btn.title       = inFS ? 'Exit full screen' : 'Full screen';
        });
        // Wait for the browser to finish the transition before resizing
        setTimeout(fitPlots, 150);
    }

    document.addEventListener('fullscreenchange',       onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', onFullscreenChange);  // Safari

    // Also close on Escape (belt-and-suspenders — browser already does this,
    // but we make sure icons get reset)
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && document.fullscreenElement) {
            document.exitFullscreen();
        } else if (e.key === 'Escape' && expanded()) {
            setExpanded(null);
        }
    });

    /**
     * MutationObserver — catches every graph Dash/React adds to the DOM.
     * Plotly adds the .js-plotly-plot class *after* the graph's div is
     * inserted, so scanning only the added nodes never found a plot and no
     * button was ever added. Instead, rescan the whole page (debounced) after
     * any change; addExpandBtn is idempotent, so repeats are free.
     */
    var pending = null;
    var observer = new MutationObserver(function () {
        if (pending) return;
        pending = setTimeout(function () {
            pending = null;
            scan(document.body);
        }, 120);
    });

    function init() {
        observer.observe(document.body, {
            childList: true, subtree: true,
            attributes: true, attributeFilter: ['class'],
        });
        scan(document.body);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

}());

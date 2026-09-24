/**
 * radar_reveal.js — SDG DISC Dashboard
 *
 * Scroll-triggered entrance for cards marked .scroll-reveal (the Graph
 * Shift Radar on the Individual Results tab):
 *   1. the card rises and fades in (CSS, see style.css "SCROLL REVEAL");
 *   2. each radar line grows out from the centre of the chart to its
 *      scores, so the Public / Stress / Mirror shifts draw themselves.
 *
 * The lines replay each time the card scrolls back into view and whenever
 * a new participant's figure arrives. Reduced-motion users get the final
 * chart with no animation.
 */
(function () {
    'use strict';

    var DURATION = 1100;             // ms for the lines to reach their scores
    var reduceMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function ease(t) { return 1 - Math.pow(1 - t, 3); }   // ease-out cubic

    /** Innermost radial value of a polar chart: where the lines start. */
    function centre(gd) {
        var ax = gd.layout && gd.layout.polar && gd.layout.polar.radialaxis;
        return ax && ax.range ? ax.range[0] : 0;
    }

    function polarTraces(gd) {
        return (gd.data || []).map(function (t, i) { return i; })
            .filter(function (i) { return gd.data[i].type === 'scatterpolar'; });
    }

    /** Collapse the lines to the centre and remember where they belong. */
    function collapse(gd) {
        // Already collapsed: re-reading r now would store the centre values
        // as the target and the lines could never grow back.
        if (gd._sdgCollapsed) return;
        var idx = polarTraces(gd);
        if (!idx.length) return;
        var c = centre(gd);
        gd._sdgTarget = idx.map(function (i) { return gd.data[i].r.slice(); });
        gd._sdgIdx = idx;
        gd._sdgBusy = true;
        Plotly.restyle(gd, {
            r: gd._sdgTarget.map(function (r) { return r.map(function () { return c; }); })
        }, idx).then(function () { gd._sdgBusy = false; });
        gd._sdgCollapsed = true;
    }

    /** Tween the collapsed lines out to their stored scores. */
    function grow(gd) {
        if (!gd._sdgCollapsed || !gd._sdgTarget) return;
        gd._sdgCollapsed = false;
        var c = centre(gd), start = null, target = gd._sdgTarget, idx = gd._sdgIdx;
        gd._sdgBusy = true;
        function frame(ts) {
            if (start === null) start = ts;
            var k = ease(Math.min(1, (ts - start) / DURATION));
            // The last frame writes the exact scores: c + (v - c) * 1 can
            // differ from v by a rounding hair, which would read as a new
            // figure and restart the animation.
            Plotly.restyle(gd, {
                r: k < 1 ? target.map(function (r) {
                    return r.map(function (v) { return c + (v - c) * k; });
                }) : target.map(function (r) { return r.slice(); })
            }, idx);
            if (k < 1) {
                requestAnimationFrame(frame);
            } else {
                gd._sdgBusy = false;
            }
        }
        requestAnimationFrame(frame);
    }

    /** Fullscreen or window-filling (fullscreen.js) takes the card out of
     *  the page flow, which reads as "scrolled away" — not a reason to
     *  collapse the lines. */
    function isEnlarged(card) {
        var fs = document.fullscreenElement || document.webkitFullscreenElement;
        return !!(card.querySelector('.graph-expanded') ||
                  (fs && (card.contains(fs) || fs.contains(card))));
    }

    function inView(el) {
        var r = el.getBoundingClientRect();
        return r.top < window.innerHeight * 0.85 && r.bottom > 0;
    }

    /** Wire one .scroll-reveal card (idempotent). */
    function wire(card) {
        if (card.dataset.revealReady) return;
        card.dataset.revealReady = '1';

        if (reduceMotion || !('IntersectionObserver' in window)) {
            card.classList.add('is-visible');
            return;
        }

        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                var gd = card.querySelector('.js-plotly-plot');
                if (e.isIntersecting) {
                    card.classList.add('is-visible');
                    if (gd) grow(gd);
                } else if (gd && !gd._sdgBusy && !isEnlarged(card)) {
                    collapse(gd);          // replay next time it comes back
                }
            });
        }, { threshold: 0.3 });
        io.observe(card);

        // Every new figure (new participant, toggled graphs, theme change)
        // starts collapsed, then grows if the card is on screen.
        function hook() {
            var gd = card.querySelector('.js-plotly-plot');
            if (!gd || gd._sdgHooked || !gd.on) return !!(gd && gd._sdgHooked);
            gd._sdgHooked = true;
            gd.on('plotly_afterplot', function () {
                if (gd._sdgBusy || gd._sdgCollapsed) return;
                var sig = JSON.stringify((gd.data || []).map(function (t) { return t.r; }));
                if (sig === gd._sdgSig) return;          // resize, hover, etc.
                gd._sdgSig = sig;
                collapse(gd);
                if (inView(card)) grow(gd);
            });
            return true;
        }
        if (!hook()) {
            var mo = new MutationObserver(function () { if (hook()) mo.disconnect(); });
            mo.observe(card, { childList: true, subtree: true });
        }
    }

    function scan(root) {
        if (!root || root.nodeType !== 1) return;
        if (root.classList && root.classList.contains('scroll-reveal')) wire(root);
        if (root.querySelectorAll) root.querySelectorAll('.scroll-reveal').forEach(wire);
    }

    function init() {
        scan(document.body);
        new MutationObserver(function (muts) {
            muts.forEach(function (m) { m.addedNodes.forEach(scan); });
        }).observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}());

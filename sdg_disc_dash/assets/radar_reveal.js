/**
 * radar_reveal.js — SDG DISC Dashboard
 *
 * Scroll-triggered entrance for cards marked .scroll-reveal (the Graph
 * Shift Radar on the Individual Results tab):
 *   1. the card rises and fades in (CSS, see style.css "SCROLL REVEAL");
 *   2. the radar lines grow out from the centre of the chart to their
 *      scores, so the Public / Stress / Mirror shifts draw themselves.
 *
 * The lines grow by scaling the chart's trace layer about the polar centre
 * with an SVG transform. The radial axis is linear and its minimum sits at
 * the centre, so scaling by k draws every point exactly where a score of
 * centre + (score - centre) * k would be — the same picture as tweening the
 * data, without asking Plotly to redraw the whole chart (grid, icons,
 * labels) on every frame. The earlier data tween did ~66 full redraws per
 * run and froze the page for seconds on each checklist click.
 *
 * The lines replay when the card scrolls back into view and when a
 * different participant's figure arrives — not when graphs are toggled on
 * or off (that only changes trace visibility). Reduced-motion users get the
 * final chart with no animation.
 */
(function () {
    'use strict';

    var DURATION = 1100;             // ms for the lines to reach their scores
    var reduceMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function ease(t) { return 1 - Math.pow(1 - t, 3); }   // ease-out cubic

    /** The group holding the radar's lines, fills and markers. */
    function layer(gd) {
        return gd.querySelector('.polarlayer .frontplot .scatterlayer');
    }

    /** Scale the lines about the polar centre (k = 1: at their scores). */
    function setScale(gd, k) {
        gd._sdgK = k;
        var el = layer(gd);
        var sub = gd._fullLayout && gd._fullLayout.polar &&
                  gd._fullLayout.polar._subplot;
        if (!el) return;
        if (k >= 1 || !sub) {
            el.removeAttribute('transform');
            return;
        }
        // The trace layer's coordinates start at the polar domain's corner,
        // so the centre is (cxx, cyy) in its own space.
        var cx = sub.cxx, cy = sub.cyy;
        el.setAttribute('transform', 'translate(' + cx + ' ' + cy + ') scale(' +
                        k + ') translate(' + (-cx) + ' ' + (-cy) + ')');
    }

    function stop(gd) {
        if (gd._sdgRaf) cancelAnimationFrame(gd._sdgRaf);
        gd._sdgRaf = null;
    }

    /** Collapse the lines to the centre, ready to grow. */
    function collapse(gd) {
        stop(gd);
        setScale(gd, 0);
    }

    /** Grow the lines from wherever they are to their scores. */
    function grow(gd) {
        if (gd._sdgK === undefined || gd._sdgK >= 1) return;
        stop(gd);
        var from = gd._sdgK, start = null;
        function frame(ts) {
            if (start === null) start = ts;
            var t = Math.min(1, (ts - start) / DURATION);
            setScale(gd, from + (1 - from) * ease(t));
            gd._sdgRaf = t < 1 ? requestAnimationFrame(frame) : null;
        }
        gd._sdgRaf = requestAnimationFrame(frame);
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

    /** Identity of the figure's data: changes for a new participant (or new
     *  scores), not when lines are shown or hidden. */
    function signature(gd) {
        return JSON.stringify((gd.data || []).map(function (t) { return t.r; }));
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
                } else if (gd && !isEnlarged(card)) {
                    collapse(gd);          // replay next time it comes back
                }
            });
        }, { threshold: 0.3 });
        io.observe(card);

        function hook() {
            var gd = card.querySelector('.js-plotly-plot');
            if (!gd || gd._sdgHooked || !gd.on) return !!(gd && gd._sdgHooked);
            gd._sdgHooked = true;
            gd.on('plotly_afterplot', function () {
                var sig = signature(gd);
                if (sig !== gd._sdgSig) {          // new participant / scores
                    gd._sdgSig = sig;
                    collapse(gd);
                    if (inView(card)) grow(gd);
                } else if (gd._sdgK !== undefined && gd._sdgK < 1) {
                    // Any redraw (toggle, resize) mid-animation or while
                    // collapsed: keep the current scale on the fresh layer.
                    setScale(gd, gd._sdgK);
                }
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

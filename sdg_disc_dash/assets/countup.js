/**
 * countup.js — SDG DISC Dashboard
 *
 * Watches for elements with class "count-up-target" appearing in the DOM
 * (added by Dash/React when metric cards render) and runs a counting
 * animation from 0 to the value stored in data-count-to.
 *
 * Attributes expected on the target element:
 *   data-count-to    {string}  Final numeric value, e.g. "1.2300" or "-0.4500"
 *   data-count-delay {string}  Milliseconds to wait before starting (optional)
 */
(function () {
    'use strict';

    var DURATION = 2400;  // animation duration in ms
    var DECIMALS = 2;     // decimal places to display

    /** Cubic ease-out — fast start, gentle arrival */
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    /**
     * Run the counter animation on a single element.
     * Guards against re-triggering if the observer fires twice.
     */
    function animateCount(el) {
        // Already animating or finished — skip
        if (el.getAttribute('data-count-running') === '1') return;

        var target = parseFloat(el.getAttribute('data-count-to'));
        if (isNaN(target)) return;

        var delay = parseInt(el.getAttribute('data-count-delay') || '0', 10);

        // Mark as running immediately so concurrent observer calls don't fire twice
        el.setAttribute('data-count-running', '1');

        // Reset display to zero while we wait for the delay
        el.textContent = '+0.00';

        setTimeout(function () {
            var startTime = null;

            function tick(now) {
                if (!startTime) startTime = now;

                var elapsed  = now - startTime;
                var progress = Math.min(elapsed / DURATION, 1);
                var eased    = easeOutCubic(progress);
                var current  = target * eased;

                // Format: always show 2 decimal places, prefix + for positive
                var sign = current >= 0 ? '+' : '';
                el.textContent = sign + current.toFixed(DECIMALS);

                if (progress < 1) {
                    requestAnimationFrame(tick);
                } else {
                    // Lock to exact target to avoid floating-point drift
                    el.textContent = (target >= 0 ? '+' : '') + target.toFixed(DECIMALS);
                    el.removeAttribute('data-count-running');
                }
            }

            requestAnimationFrame(tick);
        }, delay);
    }

    /**
     * Scan a root element (or its subtree) for any
     * .count-up-target elements and animate them.
     */
    function scanAndAnimate(root) {
        // The root itself might be the target
        if (root.classList && root.classList.contains('count-up-target')) {
            animateCount(root);
        }
        // Scan descendants
        if (root.querySelectorAll) {
            root.querySelectorAll('.count-up-target').forEach(animateCount);
        }
    }

    /**
     * MutationObserver — fires whenever Dash/React adds new nodes
     * to the DOM (e.g. after a callback updates metric-cards).
     */
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                if (node.nodeType === 1) {   // Element nodes only
                    scanAndAnimate(node);
                }
            });
        });
    });

    // Start observing once the DOM is ready
    function startObserver() {
        observer.observe(document.body, {
            childList: true,
            subtree:   true,
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startObserver);
    } else {
        startObserver();
    }

}());

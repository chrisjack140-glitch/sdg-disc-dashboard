/**
 * fullscreen.js — SDG DISC Dashboard
 *
 * Watches for Plotly graphs added to the DOM by Dash/React callbacks
 * and injects a small expand button onto each graph card.
 * Clicking it uses the native browser Fullscreen API so the card
 * fills the screen. Plotly is resized after the transition completes.
 */
(function () {
    'use strict';

    var ICON_EXPAND = '⛶';
    var ICON_CLOSE  = '✕';

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
        btn.textContent = ICON_EXPAND;

        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else if (card.requestFullscreen) {
                card.requestFullscreen();
            } else if (card.webkitRequestFullscreen) {   // Safari
                card.webkitRequestFullscreen();
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

    /** Resize all Plotly plots after a fullscreen transition. */
    function resizePlots() {
        if (!window.Plotly) return;
        document.querySelectorAll('.js-plotly-plot').forEach(function (el) {
            try { Plotly.Plots.resize(el); } catch (e) {}
        });
    }

    /** Update button icons when fullscreen state changes. */
    function onFullscreenChange() {
        var inFS = !!document.fullscreenElement;
        document.querySelectorAll('.graph-fs-btn').forEach(function (btn) {
            btn.textContent = inFS ? ICON_CLOSE : ICON_EXPAND;
            btn.title       = inFS ? 'Exit full screen' : 'Full screen';
        });
        // Wait for the browser to finish the transition before resizing
        setTimeout(resizePlots, 150);
    }

    document.addEventListener('fullscreenchange',       onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', onFullscreenChange);  // Safari

    // Also close on Escape (belt-and-suspenders — browser already does this,
    // but we make sure icons get reset)
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && document.fullscreenElement) {
            document.exitFullscreen();
        }
    });

    /** MutationObserver — catches every graph Dash/React adds to the DOM. */
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                if (node.nodeType === 1) scan(node);
            });
        });
    });

    function init() {
        observer.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

}());

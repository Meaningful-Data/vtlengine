/* The VTL operator map (docs/operator_map), as tool/render.py writes its pages.
 *
 * Every example and every list of a page starts closed. A link to one of them, such as
 * "#ex-ds-ds-manual-ex-1", opens it and the lists around it, on loading the page and on
 * following a link inside it. Only the blocks of the map (om-...) are opened. */
(function () {
  function openTarget() {
    var id = decodeURIComponent(window.location.hash.slice(1));
    var node = id ? document.getElementById(id) : null;
    for (; node; node = node.parentElement) {
      if (node.tagName === "DETAILS" && /(^| )om-/.test(node.className)) {
        node.open = true;
      }
    }
  }
  window.addEventListener("hashchange", openTarget);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", openTarget);
  } else {
    openTarget();
  }
})();

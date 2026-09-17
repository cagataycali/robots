// Two small page enhancements the stylesheet cannot express on its own.
//
// 1. A table of three or more columns stacks into one card per row on a phone
//    (see the data-sr-stack rules in docs/stylesheets/extra.css). Each cell needs to
//    carry its own column header, because CSS cannot read another element's
//    text, so the header row is copied into data-th here.
// 2. Content images load lazily unless the page already said otherwise.
// 3. The theme's search overlay is a role="dialog" with no accessible name,
//    which screen readers announce as an unnamed dialog; name it here.
(function () {
  function enhance() {
    document.querySelectorAll(".md-typeset table:not([class])").forEach(function (table) {
      var head = table.querySelector("thead tr");
      if (!head || head.children.length < 3 || table.hasAttribute("data-sr-stack")) return;
      var labels = Array.prototype.map.call(head.children, function (th) {
        return th.textContent.trim();
      });
      table.querySelectorAll("tbody tr").forEach(function (row) {
        Array.prototype.forEach.call(row.children, function (cell, i) {
          if (labels[i]) cell.setAttribute("data-th", labels[i]);
        });
      });
      // A data attribute, not a class: Material styles tables as
      // ``table:not([class])``, which a class of ours would switch off.
      table.setAttribute("data-sr-stack", "");
    });
    document.querySelectorAll(".md-content img:not([loading])").forEach(function (img) {
      img.loading = "lazy";
    });
    document.querySelectorAll('[role="dialog"]:not([aria-label])').forEach(function (dialog) {
      dialog.setAttribute("aria-label", "Search");
    });
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(enhance);
  } else {
    document.addEventListener("DOMContentLoaded", enhance);
  }
})();

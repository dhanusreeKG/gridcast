"""Builds the single-file dashboard (gridcast_live.html) from web/template.html + web/data.json + images."""
import base64, os
rd = lambda p: open(p, encoding="utf-8").read()
img = lambda p: "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
t = rd("web/template.html").replace("__DATA__", rd("web/data.json"))
t = t.replace("__IMG_A__", img("web/img/hero.jpg")).replace("__IMG_B__", img("web/img/banner.jpg"))
cdn = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
if os.path.exists("web/chart.umd.js"):          # inline Chart.js so the page works without internet
    t = t.replace(cdn, "<script>" + rd("web/chart.umd.js") + "</script>")
open("gridcast_live.html", "w", encoding="utf-8").write(t)
print("Built gridcast_live.html", round(len(t)/1024), "KB")

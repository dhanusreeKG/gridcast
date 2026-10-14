"""Official academic calendar, ODD semester 2026-27 (B.E/B.Tech Sem 3,5,7 & M.E Sem 3), transcribed from the college PDF/photo.
Types: W working day, WS working Saturday, SH Sunday holiday, H holiday, V vacation/no class,
IT1/IT2 internal test, EP ESE practical, ET ESE theory."""
import pandas as pd
D = pd.Timestamp
NAMED = {"2026-06-26":"Muharram","2026-08-15":"Independence Day","2026-08-26":"Onam","2026-09-04":"Krishna Jayanthi",
 "2026-09-14":"Vinayakar Chaturthi","2026-10-02":"Gandhi Jayanthi","2026-10-19":"Ayudha Pooja","2026-10-20":"Vijayadashami","2026-11-08":"Deepavali"}
OTHER_HOLIDAY_SAT = ["2026-07-04","2026-07-18","2026-08-01","2026-08-29","2026-09-05","2026-09-26","2026-10-03","2026-10-17","2026-10-31","2026-11-07"]
WORKING_SAT = ["2026-06-27","2026-07-11","2026-07-25","2026-08-08","2026-08-22","2026-09-12","2026-09-19","2026-10-10","2026-10-24"]
def rng(a,b): return [str(x.date()) for x in pd.date_range(a,b)]
IT1 = rng("2026-08-18","2026-08-22")
IT2 = ["2026-10-22","2026-10-23","2026-10-24","2026-10-26","2026-10-27"]
EP = ["2026-10-29","2026-10-30"] + rng("2026-11-02","2026-11-06")
ET = [d for a,b in [("2026-11-10","2026-11-14"),("2026-11-16","2026-11-21"),("2026-11-23","2026-11-28")] for d in rng(a,b)]
NO_CLASS_WEEKDAYS = ["2026-10-28","2026-11-09","2026-11-30"]    # blank in the calendar

# phase codes used by the model: 0 vacation/no class, 1 teaching, 2 internal test, 3 ESE practical, 4 ESE theory
def day_info(d):
    s = str(d.date()); wd = d.weekday()
    if d < D("2026-06-24"): return dict(type="V", label="Vacation", phase=0, holiday=int(wd==6))
    if wd == 6 and s not in ET and s not in NAMED: return dict(type="SH", label="Sunday holiday", phase=0, holiday=1)
    if s in NAMED: return dict(type="H", label=NAMED[s], phase=0, holiday=1)
    if s in OTHER_HOLIDAY_SAT: return dict(type="H", label="Holiday", phase=0, holiday=1)
    if s in IT1: return dict(type="IT1", label="Internal Test I", phase=2, holiday=0)
    if s in IT2: return dict(type="IT2", label="Internal Test II", phase=2, holiday=0)
    if s in EP: return dict(type="EP", label="ESE Practical", phase=3, holiday=0)
    if s in ET: return dict(type="ET", label="ESE Theory", phase=4, holiday=0)
    if s in NO_CLASS_WEEKDAYS: return dict(type="V", label="No class", phase=0, holiday=0)
    if wd == 5: return (dict(type="WS", label="Working Saturday", phase=1, holiday=0) if s in WORKING_SAT
                        else dict(type="H", label="Holiday", phase=0, holiday=1))
    if s == "2026-06-24": return dict(type="W", label="Reopening", phase=1, holiday=0)
    return dict(type="W", label="Working day", phase=1, holiday=0)

def proxy_info(d):
    """Typical-year calendar for years before 2026-27 (used only to build training history)."""
    md = d.month*100+d.day; wd = d.weekday()
    if wd == 6: return dict(phase=0, holiday=1)
    if md in (101,114,115,116,126,501,815,1002,1225,1020,1021,1108,904,826): return dict(phase=0, holiday=1)
    if wd == 5 and (d.isocalendar()[1] % 2): return dict(phase=0, holiday=1)
    if 108 <= md < 409: ph = 2 if 217 <= md <= 221 else 1
    elif 409 <= md <= 415: ph = 3
    elif 416 <= md <= 430: ph = 4
    elif md < 624: ph = 0
    elif md < 1029: ph = 2 if (818 <= md <= 822 or 1022 <= md <= 1027) else 1
    elif md <= 1106: ph = 3
    elif md <= 1128: ph = 4
    else: ph = 0
    return dict(phase=ph, holiday=0)

def calendar_frame(index):
    days = pd.DatetimeIndex(index.normalize().unique())
    rows = {}
    for d in days:
        rows[d] = day_info(d) if d >= D("2026-01-01") and d <= D("2026-11-30") and d >= D("2026-06-01") else proxy_info(d)
    c = pd.DataFrame.from_dict(rows, orient="index")[["holiday","phase"]]
    return c.reindex(index.normalize()).set_axis(index)

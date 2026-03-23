#%%
from ecmwfapi import ECMWFService

server = ECMWFService("mars")
print("ok")
server.execute(
    {
        "class": "rd",
        "type": "an",
        "stream": "oper",
        "expver": "ilrf",
        "levtype": "sfc",
        "param": "tcno2",
        "date": "2025-06-01",
        "time": "00:00:00",
        "area": "10/-20/-10/20",
        "grid": "1.0/1.0",
        "format": "grib",
        "expect": "any",
    },
    "test.grib",
)
# %%

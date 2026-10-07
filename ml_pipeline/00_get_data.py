"""Download freMTPL2freq (French motor third-party liability policies, CASdatasets) and save it as parquet."""
import pathlib
import urllib.request

import rdata

URL = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2freq.rda"
raw = pathlib.Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
rda = raw / "freMTPL2freq.rda"
if not rda.exists():
    urllib.request.urlretrieve(URL, rda)
frames = rdata.conversion.convert(rdata.parser.parse_file(str(rda)))
df = frames["freMTPL2freq"]
print(df.shape)
df.to_parquet(raw / "freMTPL2freq.parquet")

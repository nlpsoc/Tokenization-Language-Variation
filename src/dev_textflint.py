import textflint
from textflint import Engine
engine = Engine()
engine.run("../data/GLUE_textflint/sst2_val.csv", "../data/GLUE_textflint/config.json")
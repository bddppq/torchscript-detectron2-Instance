Small demo of writting classes that are "dynamic" before scripting but are actually "static" in the exported TorchScript.

`core.instance.Instance`: The class can dynamically add new fields (before the model being scripted).

app1 and app2 are applications built on top off core, each of them adds their customized fields into the core `Instance` class.

Each application uses `core.instance.register_fields({"n": "t"})` to add fields (in this case, the new field is called "n" and its type is "t"). After registration, getter (`get_n`) and setter (`set_n`)  will be automatically added to the `Instance` class.

To give it a try, run app1/main.py and app2/main.py.
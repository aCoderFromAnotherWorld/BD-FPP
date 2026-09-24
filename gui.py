import os
import base64
import pickle
import re
import tkinter as tk
from datetime import date, timedelta
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import javaobj.v2 as javaobj

# Replace this string with the actual path to your Java installation
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-26.0.1"

# ======================================================================
# SECTION 1: Read a .model file's own training-time schema.
# ======================================================================

SKLEARN_CLASSNAME = "weka.classifiers.sklearn.ScikitLearnClassifier"
REMOVE_FILTER_CLASSNAME = "weka.filters.unsupervised.attribute.Remove"

def _field(obj, name):
    fd = getattr(obj, "field_data", None)
    if not fd: return None
    for _classdesc, fields in fd.items():
        for fname, fval in fields.items():
            if getattr(fname, "name", None) == name:
                return fval
    return None

def _nominal_categories(attribute_info_obj):
    values_field = _field(attribute_info_obj, "m_Values")
    if values_field is None: return None
    anno = getattr(values_field, "annotations", None)
    if not anno: return None
    for classdesc, alist in anno.items():
        if classdesc.name == "java.util.ArrayList":
            return [str(v) for v in alist if isinstance(v, (javaobj.beans.JavaString, str))]
    return None

def _find_nested(obj, target_classname, _seen=None):
    if _seen is None: _seen = set()
    oid = id(obj)
    if oid in _seen: return None
    _seen.add(oid)

    cd = getattr(obj, "classdesc", None)
    if cd is not None and cd.name == target_classname:
        return obj

    fd = getattr(obj, "field_data", None)
    if fd:
        for _classdesc, fields in fd.items():
            for _fname, fval in fields.items():
                if hasattr(fval, "classdesc") or hasattr(fval, "field_data"):
                    found = _find_nested(fval, target_classname, _seen)
                    if found is not None: return found
                elif isinstance(fval, (list, tuple)):
                    for item in fval:
                        if hasattr(item, "classdesc") or hasattr(item, "field_data"):
                            found = _find_nested(item, target_classname, _seen)
                            if found is not None: return found
    return None

def _find_remove_retained_indices(clf_obj):
    remove_obj = _find_nested(clf_obj, REMOVE_FILTER_CLASSNAME)
    if remove_obj is None: return None
    retained = _field(remove_obj, "m_SelectedAttributes")
    if retained is None: return None
    return sorted(int(i) for i in retained)

def extract_model_metadata(model_path):
    with open(model_path, "rb") as fh:
        obj = javaobj.load(fh)
    clf_obj = obj[0]
    insts = obj[1]

    class_index = _field(insts, "m_ClassIndex")
    attrs = _field(insts, "m_Attributes")

    full_attr_names = []
    nominal_orders = {}
    for a in attrs:
        name = _field(a, "m_Name")
        name = str(name) if name is not None else None
        full_attr_names.append(name)
        ainfo = _field(a, "m_AttributeInfo")
        if ainfo is not None and getattr(getattr(ainfo, "classdesc", None), "name", "") == "weka.core.NominalAttributeInfo":
            cats = _nominal_categories(ainfo)
            if cats: nominal_orders[name] = cats

    sklearn_obj = _find_nested(clf_obj, SKLEARN_CLASSNAME)
    is_sklearn = sklearn_obj is not None
    sklearn_b64 = None
    if is_sklearn:
        fd = getattr(sklearn_obj, "field_data", None)
        for _classdesc, fields in fd.items():
            for _fname, fval in fields.items():
                if isinstance(fval, (javaobj.beans.JavaString, str)) and len(str(fval)) > 500:
                    sklearn_b64 = str(fval)

    input_attr_names = full_attr_names[:class_index]
    retained_indices = _find_remove_retained_indices(clf_obj)
    
    if retained_indices is not None:
        retained_set = set(retained_indices)
        sklearn_input_attr_names = [name for i, name in enumerate(input_attr_names) if i in retained_set]
    else:
        sklearn_input_attr_names = input_attr_names

    return {
        "classifier_class": clf_obj.classdesc.name,
        "is_sklearn": is_sklearn,
        "sklearn_pickle_b64": sklearn_b64,
        "full_attr_names": full_attr_names,
        "class_index": class_index,
        "class_name": full_attr_names[class_index],
        "input_attr_names": input_attr_names,
        "sklearn_input_attr_names": sklearn_input_attr_names,
        "nominal_orders": nominal_orders,
    }

# ======================================================================
# SECTION 2 & 3: Predictors
# ======================================================================

def predict_sklearn_model(model_path, df, meta=None):
    if meta is None: meta = extract_model_metadata(model_path)
    if not meta["is_sklearn"]: raise ValueError("No embedded scikit-learn estimator")

    estimator = pickle.loads(base64.b64decode(meta["sklearn_pickle_b64"]))
    nominal_orders = meta["nominal_orders"]
    input_attrs = meta.get("sklearn_input_attr_names", meta["input_attr_names"])

    n = len(df)
    skip_mask = np.zeros(n, dtype=bool)
    skip_reason = [""] * n

    blocks = []
    for col in input_attrs:
        if col in nominal_orders:
            cats = nominal_orders[col]
            cat_index = {c: i for i, c in enumerate(cats)}
            block = np.zeros((n, len(cats)), dtype=float)
            for row_i, val in enumerate(df[col].values):
                idx = cat_index.get(val)
                if idx is None:
                    skip_mask[row_i] = True
                    skip_reason[row_i] = f"unknown {col}: {val}"
                else:
                    block[row_i, idx] = 1.0
            blocks.append(block)
        else:
            blocks.append(df[[col]].to_numpy(dtype=float))
            
    X = np.hstack(blocks)
    preds = np.full(n, np.nan, dtype=float)
    valid_idx = np.where(~skip_mask)[0]
    
    if len(valid_idx) > 0:
        preds[valid_idx] = estimator.predict(X[valid_idx])

    return preds, pd.Series(skip_mask, index=df.index), pd.Series(skip_reason, index=df.index)

def predict_weka_model(model_path, df, meta=None):
    from weka.core.dataset import Attribute, Instances, Instance
    from weka.core.serialization import read as weka_read
    from weka.classifiers import Classifier

    if meta is None: meta = extract_model_metadata(model_path)
    classifier = Classifier(jobject=weka_read(model_path))

    nominal_orders = meta["nominal_orders"]
    input_attrs = meta["input_attr_names"]
    class_name = meta["class_name"]

    atts = []
    for name in input_attrs:
        if name in nominal_orders:
            atts.append(Attribute.create_nominal(name, nominal_orders[name]))
        else:
            atts.append(Attribute.create_numeric(name))
    atts.append(Attribute.create_numeric(class_name))
    
    dataset = Instances.create_instances("test", atts, 0)
    dataset.class_index = len(atts) - 1

    cat_indices = {col: {c: i for i, c in enumerate(nominal_orders[col])} for col in nominal_orders}

    n = len(df)
    preds = np.full(n, np.nan, dtype=float)
    skip_mask = np.zeros(n, dtype=bool)
    skip_reason = [""] * n

    columns = df.columns.tolist()
    for row_i, row in enumerate(df.itertuples(index=False, name=None)):
        row_d = dict(zip(columns, row))
        values = []
        bad = None
        for name in input_attrs:
            if name in nominal_orders:
                idx = cat_indices[name].get(row_d[name])
                if idx is None:
                    bad = f"unknown {name}: {row_d[name]}"
                    values.append(0.0)
                else:
                    values.append(float(idx))
            else:
                values.append(float(row_d[name]))
        values.append(float("nan"))

        if bad is not None:
            skip_mask[row_i] = True
            skip_reason[row_i] = bad
            continue

        inst = Instance.create_instance(values)
        inst.dataset = dataset
        preds[row_i] = classifier.classify_instance(inst)

    return preds, pd.Series(skip_mask, index=df.index), pd.Series(skip_reason, index=df.index)

# ======================================================================
# SECTION 4: Data Prep
# ======================================================================

LAG_RE = re.compile(r"^Price_T-(\d+)$")
MA_RE = re.compile(r"^Price_(\d+)d_MA$")

def parse_lag_spec(input_attr_names, nominal_attr_names):
    spec = {"lags": {}, "mas": {}, "date_parts": [], "nominal": [], "current_price": None, "passthrough": []}
    for name in input_attr_names:
        if name in nominal_attr_names: spec["nominal"].append(name)
        elif name in ("year", "month", "day"): spec["date_parts"].append(name)
        elif name == "average_price": spec["current_price"] = name
        else:
            m = LAG_RE.match(name)
            if m:
                spec["lags"][name] = int(m.group(1))
                continue
            m = MA_RE.match(name)
            if m:
                spec["mas"][name] = int(m.group(1))
                continue
            spec["passthrough"].append(name)
    return spec

def build_feature_row(history_asc, as_of_date, division, commodity, unit, lag_spec):
    before = history_asc[history_asc["_date"] <= pd.Timestamp(as_of_date)].sort_values("_date", ascending=False)
    if not before.empty:
        hist = before
    else:
        hist = history_asc.assign(_diff=(history_asc["_date"] - pd.Timestamp(as_of_date)).abs()) \
                           .sort_values("_diff").sort_values("_date", ascending=False)

    row = {}
    for name in lag_spec["nominal"]:
        row[name] = {"division": division, "commodity_name": commodity, "retail_unit": unit}[name]
    for part in lag_spec["date_parts"]:
        row[part] = getattr(pd.Timestamp(as_of_date), part)
    if lag_spec["current_price"]:
        row[lag_spec["current_price"]] = float(hist.iloc[0]["average_price"])
    for name, n in lag_spec["lags"].items():
        idx = min(n - 1, len(hist) - 1)
        row[name] = float(hist.iloc[idx]["average_price"])
    for name, n in lag_spec["mas"].items():
        row[name] = float(hist.head(n)["average_price"].mean())
    for name in lag_spec["passthrough"]:
        row[name] = float(hist.iloc[0][name]) if name in hist.columns else 0.0
    return row

def load_reference_data(path):
    if path.lower().endswith(".arff"):
        import csv, io
        columns, data_lines, in_data = [], [], False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("%"): continue
                if in_data:
                    data_lines.append(line)
                    continue
                if line.lower().startswith("@attribute"):
                    rest = line[len("@attribute"):].strip()
                    name = rest[1:rest.index("'", 1)] if rest.startswith("'") else rest.split(None, 1)[0]
                    columns.append(name)
                elif line.lower().startswith("@data"):
                    in_data = True
        rows = [next(csv.reader(io.StringIO(l), skipinitialspace=True, quotechar="'")) for l in data_lines]
        df = pd.DataFrame(rows, columns=columns)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
    return pd.read_csv(path)

# ======================================================================
# SECTION 5: Dynamic Routing & GUI
# ======================================================================

MODEL_FAMILIES = {
    "Gradient Boosting": ("sklearn", "gbr"),
    "XGBoost": ("sklearn", "xgbr"),
    "Linear Regression": ("weka", "lr"),
    "M5P (model tree)": ("weka", "m5p"),
    "Random Forest": ("weka", "rf"),
}

class ModelFamily:
    def __init__(self, name, kind, prefix, base_dir, split_type):
        self.name = name
        self.kind = kind
        self.split_type = split_type.lower()
        self.base_dir = base_dir
        self.prefix = prefix
        self._meta = {}
        
    def get_path(self, horizon):
        filename = f"{self.prefix}_random_model_{horizon}.model" if self.split_type == "random" else f"{self.prefix}_model_{horizon}.model"
        return os.path.join(self.base_dir, self.split_type, f"target_{horizon}", filename)

    def available_horizons(self):
        return [7, 14, 30]

    def meta(self, horizon):
        if horizon not in self._meta:
            self._meta[horizon] = extract_model_metadata(self.get_path(horizon))
        return self._meta[horizon]

    def run(self, horizon, jvm_ensurer, row):
        meta = self.meta(horizon)
        df = pd.DataFrame([row])
        if self.kind == "sklearn":
            preds, skip_mask, skip_reason = predict_sklearn_model(self.get_path(horizon), df, meta=meta)
        else:
            jvm_ensurer()
            preds, skip_mask, skip_reason = predict_weka_model(self.get_path(horizon), df, meta=meta)
        if skip_mask.iloc[0]: raise ValueError(str(skip_reason.iloc[0]))
        return float(preds[0])

    def dynamic_predict(self, history, division, commodity, unit, target_date, jvm_ensurer):
        max_date = history["_date"].max().date()
        steps_log = []
        
        if target_date <= max_date:
            h = 7
            row = build_feature_row(history, target_date - timedelta(days=h), division, commodity, unit, parse_lag_spec(self.meta(h)["input_attr_names"], list(self.meta(h)["nominal_orders"].keys())))
            price = self.run(h, jvm_ensurer, row)
            steps_log.append(f"Direct 7-day model, as-of {target_date - timedelta(days=h)} -> {target_date}: {price:.2f}")
            return price, steps_log

        days_needed = (target_date - max_date).days
        
        horizons_sorted = [30, 14, 7]
        plan, remaining = [], days_needed
        for h in horizons_sorted:
            while remaining >= h: plan.append(h); remaining -= h
        if remaining > 0: plan.append(7)
        
        working = history.copy()
        current = max_date

        for h in plan:
            spec = parse_lag_spec(self.meta(h)["input_attr_names"], list(self.meta(h)["nominal_orders"].keys()))
            row = build_feature_row(working, current, division, commodity, unit, spec)
            price = self.run(h, jvm_ensurer, row)
            next_date = min(current + timedelta(days=h), target_date)
            steps_log.append(f"{h}-day model, as-of {current} -> {next_date}: {price:.2f}")
            
            new_row = {"_date": pd.Timestamp(next_date), "average_price": price}
            for c in working.columns:
                if c not in ("_date", "average_price"):
                    new_row[c] = working.sort_values("_date").iloc[-1][c]
            working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
            current = next_date

        if current < target_date:
            h = 7
            row = build_feature_row(working, current, division, commodity, unit, parse_lag_spec(self.meta(h)["input_attr_names"], list(self.meta(h)["nominal_orders"].keys())))
            price = self.run(h, jvm_ensurer, row)
            steps_log.append(f"7-day model (final gap), as-of {current} -> {target_date}: {price:.2f}")

        return price, steps_log

class PricePredictorApp:
    def __init__(self, root, default_data_path="dataset.csv"):
        self.root = root
        self.jvm_started = False
        self.base_model_dir = "Model" 
        self.default_data_path = default_data_path
        
        root.title("Dynamic Price Predictor")
        root.geometry("600x680")

        self.ref_df = None
        pad = {"padx": 10, "pady": 4}

        # Control Frame
        frame_ctrl = ttk.LabelFrame(root, text="System Config")
        frame_ctrl.pack(fill="x", **pad)
        
        self.lbl_ref_status = ttk.Label(frame_ctrl, text="Status: Initializing Dataset...", foreground="orange")
        self.lbl_ref_status.pack(side="left", padx=10, pady=10)

        # Model Config Frame
        frame_model = ttk.LabelFrame(root, text="Model Configuration")
        frame_model.pack(fill="x", **pad)
        
        ttk.Label(frame_model, text="Split:").grid(row=0, column=0, padx=5, pady=5)
        self.split_var = tk.StringVar(value="Temporal")
        ttk.Combobox(frame_model, textvariable=self.split_var, values=["Temporal", "Random"], state="readonly", width=15).grid(row=0, column=1)

        ttk.Label(frame_model, text="Algorithm:").grid(row=0, column=2, padx=5, pady=5)
        self.family_var = tk.StringVar(value=list(MODEL_FAMILIES.keys())[0])
        ttk.Combobox(frame_model, textvariable=self.family_var, values=list(MODEL_FAMILIES.keys()), state="readonly", width=25).grid(row=0, column=3)

        # Target Specs
        frame_cat = ttk.LabelFrame(root, text="Product to Predict")
        frame_cat.pack(fill="x", **pad)
        self.division_var, self.commodity_var, self.unit_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.commodity_units = {}

        self._labeled_combo(frame_cat, "Division:", self.division_var, "division_combo")
        self._labeled_combo(frame_cat, "Commodity:", self.commodity_var, "commodity_combo")
        self.commodity_combo.bind("<<ComboboxSelected>>", lambda e: self._update_unit_for_commodity())
        self._labeled_combo(frame_cat, "Retail unit:", self.unit_var, "unit_combo")

        # Date to Predict
        frame_date = ttk.LabelFrame(root, text="Target Date to Predict")
        frame_date.pack(fill="x", **pad)
        today = date.today()
        self.year_var, self.month_var, self.day_var = tk.IntVar(value=today.year), tk.IntVar(value=today.month), tk.IntVar(value=today.day)
        
        row = ttk.Frame(frame_date)
        row.pack(padx=6, pady=6)
        ttk.Label(row, text="YYYY:").grid(row=0, column=0); ttk.Spinbox(row, from_=2000, to=2100, textvariable=self.year_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(row, text="MM:").grid(row=0, column=2); ttk.Spinbox(row, from_=1, to=12, textvariable=self.month_var, width=4).grid(row=0, column=3, padx=4)
        ttk.Label(row, text="DD:").grid(row=0, column=4); ttk.Spinbox(row, from_=1, to=31, textvariable=self.day_var, width=4).grid(row=0, column=5, padx=4)

        ttk.Button(root, text="Predict Price", command=self.predict).pack(pady=10)

        # Output Text
        frame_result = ttk.LabelFrame(root, text="Prediction Log & Result")
        frame_result.pack(fill="both", expand=True, **pad)
        self.result_text = tk.Text(frame_result, height=12, wrap="word", state="disabled")
        self.result_text.pack(fill="both", expand=True, padx=6, pady=6)

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Trigger auto-load slightly after UI renders
        self.root.after(100, self.auto_load_reference)

    def _labeled_combo(self, parent, label, var, attr_name):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=12).pack(side="left")
        combo = ttk.Combobox(row, textvariable=var, state="readonly", width=35)
        combo.pack(side="left", fill="x", expand=True)
        setattr(self, attr_name, combo)

    def _set_result(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def auto_load_reference(self):
        if not os.path.exists(self.default_data_path):
            self.lbl_ref_status.config(text=f"Status: Missing '{os.path.basename(self.default_data_path)}'", foreground="red")
            messagebox.showwarning(
                "Missing Data", 
                f"Could not find the default dataset:\n{self.default_data_path}\n\nPlease ensure the file exists in the directory."
            )
            return

        try:
            self.ref_df = load_reference_data(self.default_data_path)
            self.lbl_ref_status.config(text=f"Status: Data Loaded ({os.path.basename(self.default_data_path)})", foreground="green")
        except Exception as e:
            self.lbl_ref_status.config(text="Status: Error loading data", foreground="red")
            messagebox.showerror("Error", f"Failed to auto-load data:\n{e}")
            return

        divs = sorted(self.ref_df["division"].dropna().unique().tolist())
        coms = sorted(self.ref_df["commodity_name"].dropna().unique().tolist())
        self.commodity_units = {c: sorted(g["retail_unit"].dropna().unique().tolist()) for c, g in self.ref_df.groupby("commodity_name")}

        self.division_combo["values"] = divs
        self.commodity_combo["values"] = coms
        if divs: self.division_var.set(divs[0])
        if coms: self.commodity_var.set(coms[0])
        self._update_unit_for_commodity()

    def _update_unit_for_commodity(self):
        units = self.commodity_units.get(self.commodity_var.get(), [])
        self.unit_combo["values"] = units
        self.unit_var.set(units[0] if units else "")

    def _ensure_jvm(self):
        if not self.jvm_started:
            import weka.core.jvm as jvm
            jvm.start(packages=True)
            self.jvm_started = True

    def on_close(self):
        if self.jvm_started:
            try:
                import weka.core.jvm as jvm
                jvm.stop()
            except Exception: pass
        self.root.destroy()

    def predict(self):
        if self.ref_df is None:
            messagebox.showerror("Error", "Reference Data was not loaded successfully.")
            return

        name = self.family_var.get()
        kind, prefix = MODEL_FAMILIES[name]
        family = ModelFamily(name, kind, prefix, self.base_model_dir, self.split_var.get())

        for h in family.available_horizons():
            if not os.path.exists(family.get_path(h)):
                messagebox.showerror("Missing Model", f"Required model file not found:\n{family.get_path(h)}")
                return

        div, com, unit = self.division_var.get(), self.commodity_var.get(), self.unit_var.get()
        try: target = date(self.year_var.get(), self.month_var.get(), self.day_var.get())
        except ValueError as e: return messagebox.showerror("Date", f"Invalid date: {e}")

        df = self.ref_df.copy()
        df["_date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")
        history = df[(df["division"] == div) & (df["commodity_name"] == com) & (df["retail_unit"] == unit)].dropna(subset=["_date"]).sort_values("_date")
        
        if history.empty:
            return messagebox.showerror("Error", "No historical data found for this combination.")

        self._set_result("Predicting... (This may take a moment if launching JVM)")
        self.root.update_idletasks()

        try:
            price, steps = family.dynamic_predict(history, div, com, unit, target, self._ensure_jvm)
            
            lines = [
                f"Model: {name} | Split: {self.split_var.get()}",
                f"Product: {div} / {com} / {unit}",
                f"Target Forecast Date: {target}\n",
                "Dynamic Jump Log:"
            ]
            for s in steps: lines.append(f"  - {s}")
            lines.append(f"\n==> FINAL PREDICTED PRICE: {price:.2f}")
            self._set_result("\n".join(lines))
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._set_result("")

if __name__ == "__main__":
    # UPDATE THIS CONSTANT TO YOUR LOCAL FILENAME/PATH (e.g., 'dataset.csv' or 'data.arff')
    DEFAULT_DATA_PATH = "Preprocessing/moa_train_80_lag.csv"  # Change this to your actual dataset path
    
    root = tk.Tk()
    app = PricePredictorApp(root, default_data_path=DEFAULT_DATA_PATH)
    root.mainloop()
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, ttest_1samp, chi2_contingency, binom as binom_dist, poisson
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import warnings, random

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Load & clean data ──────────────────────────
def load():
    import os
    # Works whether you run from the DAL folder or anywhere else
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "medical_data.csv")
    df = pd.read_csv(csv_path)

    # Strip column name whitespace FIRST, then filter
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    # Drop blank rows (CSV has empty rows between groups)
    df = df[df["Name"].notna() & df["Name"].ne("")]

    df["Disease"]  = df["Disease"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["Symptoms"] = df["Symptoms"].fillna("")
    df["Causes"]   = df["Causes"].fillna("Unknown")
    df["Medicine"] = df["Medicine"].fillna("Consult Doctor")
    df["Gender"]   = df["Gender"].fillna("Unknown")

    def age(dob):
        for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"]:
            try: return max(1, 2025 - datetime.strptime(str(dob).strip(), fmt).year)
            except: pass
        return None

    df["Age"] = df["DateOfBirth"].apply(age)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df

DF = load()

def cosine(a, b):
    sa = {x.lower().strip() for x in a}
    sb = {x.lower().strip() for x in b}
    all_t = sa | sb
    if not all_t: return 0.0
    va = np.array([1 if t in sa else 0 for t in all_t], float)
    vb = np.array([1 if t in sb else 0 for t in all_t], float)
    d  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va,vb)/d) if d else 0.0

def fuzzy(qa, ra):
    score = 0.0
    for q in qa:
        for r in ra:
            ql,rl = q.lower(), r.lower()
            if ql==rl: score+=1.0
            elif ql in rl or rl in ql: score+=0.7
            else: score += len(set(ql)&set(rl))/(max(len(ql),len(rl))+1)*0.3
    return score/max(len(qa),1)

# ── Serve index.html ──
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── Diagnose ──
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    body = request.json
    syms = body.get("symptoms", [])
    algo = body.get("algo", "cf")
    eff  = [s for s in syms if random.random()>0.05] if algo=="pp" else syms
    rows = []
    for _, row in DF.iterrows():
        rs = [s.strip() for s in str(row["Symptoms"]).split(",")]
        sim = (0.6*fuzzy(eff,rs)+0.4*cosine(eff,rs)) if algo=="hybrid" else cosine(eff,rs)
        if sim > 0:
            rows.append({"disease":row["Disease"],"symptoms":row["Symptoms"],
                         "medicine":row["Medicine"],"causes":row["Causes"],
                         "age":int(row["Age"]) if pd.notna(row["Age"]) else None,
                         "gender":row["Gender"],"sim":round(sim,4)})
    rows.sort(key=lambda x:x["sim"], reverse=True)
    return jsonify({"results": rows[:8]})

# ── Exp 1: Box Plot ──
@app.route("/api/exp1")
def exp1():
    dis  = request.args.get("disease","")
    sub  = DF[DF["Disease"].str.contains(dis, case=False, na=False)] if dis else DF
    ages = sub["Age"].dropna().values
    if len(ages)<5: ages = DF["Age"].dropna().values
    q1,q3 = np.percentile(ages,25), np.percentile(ages,75)
    iqr   = q3-q1
    out   = ages[(ages<q1-1.5*iqr)|(ages>q3+1.5*iqr)].tolist()
    gd = {}
    for g in ["Male","Female"]:
        ga = DF[DF["Gender"]==g]["Age"].dropna().values
        if len(ga): gd[g]={"min":float(np.min(ga)),"q1":float(np.percentile(ga,25)),
            "median":float(np.median(ga)),"q3":float(np.percentile(ga,75)),"max":float(np.max(ga))}
    bins=12; bw=(ages.max()-ages.min())/bins
    counts=[int(((ages>=ages.min()+i*bw)&(ages<ages.min()+(i+1)*bw)).sum()) for i in range(bins)]
    labels=[f"{int(ages.min()+i*bw)}" for i in range(bins)]
    return jsonify({"q1":round(q1,1),"q3":round(q3,1),"median":round(float(np.median(ages)),1),
        "iqr":round(iqr,1),"mean":round(float(ages.mean()),1),"std":round(float(ages.std()),1),
        "min":round(float(ages.min()),1),"max":round(float(ages.max()),1),
        "outliers":out,"n":len(ages),"gender":gd,
        "hist_counts":counts,"hist_labels":labels,
        "lower_fence":round(q1-1.5*iqr,1),"upper_fence":round(q3+1.5*iqr,1)})

# ── Exp 2: Regression ──
@app.route("/api/exp2")
def exp2():
    df2 = DF.dropna(subset=["Age"]).copy()
    df2["symc"] = df2["Symptoms"].apply(lambda s: len(str(s).split(",")))
    X = df2[["Age"]].values; y = df2["symc"].values
    m = LinearRegression().fit(X,y)
    yp= m.predict(X)
    r2= r2_score(y,yp)
    idx = np.random.RandomState(42).choice(len(df2), min(80,len(df2)), replace=False)
    scatter=[{"x":int(df2.iloc[i]["Age"]),"y":int(y[i])} for i in idx]
    ar = list(range(int(X.min()),int(X.max())+1,2))
    pl = [round(float(m.predict([[a]])[0]),2) for a in ar]
    resid=[round(float(y[i]-yp[i]),2) for i in idx]
    return jsonify({"scatter":scatter,"age_range":ar,"pred_line":pl,
        "slope":round(float(m.coef_[0]),4),"intercept":round(float(m.intercept_),4),
        "r2":round(r2,4),"n":len(df2),"residuals":resid,
        "equation":f"SymCount = {round(float(m.coef_[0]),4)} × Age + {round(float(m.intercept_),4)}"})

# ── Exp 3: Sampling ──
@app.route("/api/exp3")
def exp3():
    method=request.args.get("method","random"); k=int(request.args.get("k",30))
    df3=DF.dropna(subset=["Age"]).reset_index(drop=True)
    if method=="random": smp=df3.sample(min(k,len(df3)),random_state=42)
    else:
        step=max(1,len(df3)//k); smp=df3.iloc[list(range(0,len(df3),step))[:k]]
    pa=df3["Age"].tolist(); sa=smp["Age"].tolist()
    return jsonify({"pop_n":len(df3),"smp_n":len(smp),
        "pop_mean":round(float(np.mean(pa)),2),"smp_mean":round(float(np.mean(sa)),2),
        "pop_std":round(float(np.std(pa)),2),"smp_std":round(float(np.std(sa)),2),
        "pop_ages":sorted(pa),"smp_ages":sorted(sa),
        "pop_dist":df3["Disease"].value_counts().head(8).to_dict(),
        "smp_dist":smp["Disease"].value_counts().head(8).to_dict()})

# ── Exp 4: Clustering ──
@app.route("/api/exp4")
def exp4():
    k=int(request.args.get("k",3))
    df4=DF.dropna(subset=["Age"]).copy()
    df4["symc"]=df4["Symptoms"].apply(lambda s:len(str(s).split(",")))
    X=df4[["Age","symc"]].values
    km=KMeans(n_clusters=k,random_state=42,n_init=10); df4["cl"]=km.fit_predict(X)
    inertias=[]
    for ki in range(2,8):
        km_i=KMeans(n_clusters=ki,random_state=42,n_init=10); km_i.fit(X)
        inertias.append(round(float(km_i.inertia_),1))
    clusters=[{"id":i,"size":int((df4["cl"]==i).sum()),
        "cx":round(float(km.cluster_centers_[i][0]),1),
        "cy":round(float(km.cluster_centers_[i][1]),1),
        "top":df4[df4["cl"]==i]["Disease"].value_counts().head(3).to_dict()} for i in range(k)]
    scatter=[{"x":int(r["Age"]),"y":int(r["symc"]),"c":int(r["cl"])} for _,r in df4.iterrows()]
    return jsonify({"k":k,"scatter":scatter,"clusters":clusters,
        "elbow_ks":list(range(2,8)),"inertias":inertias})

# ── Exp 5: Probability ──
@app.route("/api/exp5")
def exp5():
    ages=DF["Age"].dropna().values; mu,sig=float(ages.mean()),float(ages.std())
    x=np.linspace(mu-3*sig,mu+3*sig,80)
    dis_p=DF["Disease"].value_counts(); total=len(DF)
    prob=[{"disease":d,"p":round(c/total,4)} for d,c in dis_p.head(10).items()]
    bk=list(range(5)); bpmf=[round(float(binom_dist.pmf(k,4,0.4)),4) for k in bk]
    lam=float(dis_p.mean()); pk=list(range(12))
    ppmf=[round(float(poisson.pmf(k,lam)),4) for k in pk]
    return jsonify({"mu":round(mu,1),"sigma":round(sig,1),
        "x":[round(v,1) for v in x],"pdf":[round(float(norm.pdf(v,mu,sig)),5) for v in x],
        "cdf":[round(float(norm.cdf(v,mu,sig)),5) for v in x],
        "prob_table":prob,"binom":{"ks":bk,"pmf":bpmf,"n":4,"p":0.4},
        "poisson":{"ks":pk,"pmf":ppmf,"lambda":round(lam,2)}})

# ── Exp 6: Stat Properties ──
@app.route("/api/exp6")
def exp6():
    ages=DF["Age"].dropna().values
    mode_r=stats.mode(ages,keepdims=True)
    hist,edges=np.histogram(ages,bins=15)
    centers=[(edges[i]+edges[i+1])/2 for i in range(len(hist))]
    gst={}
    for g in ["Male","Female"]:
        ga=DF[DF["Gender"]==g]["Age"].dropna().values
        if len(ga): gst[g]={"mean":round(float(ga.mean()),2),"median":round(float(np.median(ga)),2),
            "std":round(float(ga.std()),2),"skew":round(float(stats.skew(ga)),3),
            "kurt":round(float(stats.kurtosis(ga)),3),"n":len(ga)}
    return jsonify({"mean":round(float(ages.mean()),2),"median":round(float(np.median(ages)),2),
        "mode":round(float(mode_r.mode[0]),2),"std":round(float(ages.std()),2),
        "variance":round(float(ages.var()),2),"skewness":round(float(stats.skew(ages)),4),
        "kurtosis":round(float(stats.kurtosis(ages)),4),"min":round(float(ages.min()),1),
        "max":round(float(ages.max()),1),"range":round(float(ages.max()-ages.min()),1),
        "n":len(ages),"hist_counts":hist.tolist(),"hist_bins":[round(c,1) for c in centers],
        "gender_stats":gst})

# ── Exp 7: Inference ──
@app.route("/api/exp7")
def exp7():
    dis=request.args.get("disease","Influenza")
    ages=DF["Age"].dropna().values; pop_mu=float(ages.mean())
    d_ages=DF[DF["Disease"].str.contains(dis,case=False,na=False)]["Age"].dropna().values
    if len(d_ages)<3: d_ages=DF["Age"].dropna().sample(15,random_state=1).values
    t,p=ttest_1samp(d_ages,pop_mu)
    n=len(d_ages); se=stats.sem(d_ages)
    ci=stats.t.interval(0.95,df=n-1,loc=d_ages.mean(),scale=se)
    top5=DF["Disease"].value_counts().head(5).index.tolist()
    ct=pd.crosstab(DF[DF["Disease"].isin(top5)]["Gender"],DF[DF["Disease"].isin(top5)]["Disease"])
    chi2_v,p_chi,dof,_=chi2_contingency(ct)
    return jsonify({"disease":dis,"pop_mu":round(pop_mu,2),"d_mean":round(float(d_ages.mean()),2),
        "d_std":round(float(d_ages.std()),2),"d_n":n,"t_stat":round(float(t),4),
        "p_val":round(float(p),4),"reject_h0":bool(p<0.05),
        "ci_lower":round(float(ci[0]),2),"ci_upper":round(float(ci[1]),2),
        "chi2":round(float(chi2_v),4),"p_chi":round(float(p_chi),4),
        "chi_dof":int(dof),"z_scores":[round(float(z),3) for z in stats.zscore(d_ages)],
        "d_ages":d_ages.tolist(),"top5_diseases":top5,"crosstab":ct.to_dict()})

# ── Exp 8: Time Series ──
@app.route("/api/exp8")
def exp8():
    np.random.seed(42); n=36
    trend_line=np.linspace(20,45,n)
    seasonal=8*np.sin(np.linspace(0,4*np.pi,n))
    noise=np.random.normal(0,3,n)
    series=(trend_line+seasonal+noise).clip(5).round(1).tolist()
    def ma(data,w): return [None]*(w-1)+[round(float(np.mean(data[i-w+1:i+1])),2) for i in range(w-1,len(data))]
    x=np.arange(n); lr=LinearRegression().fit(x.reshape(-1,1),series)
    trend=[round(float(lr.predict([[i]])[0]),2) for i in x]
    fc_x=np.arange(n,n+6)
    fc=[round(float(lr.predict([[i]])[0])+8*np.sin((n+j)*2*np.pi/12),2) for j,i in enumerate(fc_x)]
    start=datetime(2022,1,1)
    labels=[(start+timedelta(days=30*i)).strftime("%b %Y") for i in range(n)]
    fc_labels=[(start+timedelta(days=30*(n+i))).strftime("%b %Y") for i in range(6)]
    top5=DF["Disease"].value_counts().head(5)
    quarterly={d:[int(c//4+np.random.randint(-2,3)) for _ in range(4)] for d,c in top5.items()}
    return jsonify({"labels":labels,"series":series,"ma3":ma(series,3),"ma6":ma(series,6),
        "trend":trend,"fc_labels":fc_labels,"forecast":fc,
        "slope":round(float(lr.coef_[0]),4),"r2":round(float(r2_score(series,trend)),4),
        "quarterly":quarterly,"top5":list(top5.keys())})

# ── Summary ──
@app.route("/api/summary")
def summary():
    return jsonify({"total":len(DF),"diseases":int(DF["Disease"].nunique()),
        "avg_age":round(float(DF["Age"].dropna().mean()),1),
        "top_diseases":DF["Disease"].value_counts().head(10).to_dict(),
        "top_causes":DF["Causes"].value_counts().head(6).to_dict(),
        "gender":DF["Gender"].value_counts().to_dict()})

if __name__=="__main__":
    print("\n  MediRec running → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
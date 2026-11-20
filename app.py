import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main{background:#f6f8fb}
[data-testid="stSidebar"]{background:#111827}
[data-testid="stSidebar"] *{color:white!important}
.hero{padding:26px;border-radius:18px;background:linear-gradient(135deg,#0f172a,#2563eb);color:white;margin-bottom:22px}
.hero h1{margin:0;font-size:32px}.hero p{opacity:.9}
.kpi{background:white;padding:18px;border-radius:15px;border:1px solid #e5e7eb;box-shadow:0 4px 14px #0000000d}
.kpi-label{color:#64748b;font-size:12px;font-weight:700}.kpi-value{font-size:28px;font-weight:800;margin-top:7px}
.card{background:white;padding:18px;border-radius:15px;border:1px solid #e5e7eb}
.risk-low{padding:18px;border-radius:14px;background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;font-weight:700}
.risk-medium{padding:18px;border-radius:14px;background:#fffbeb;border:1px solid #fde68a;color:#92400e;font-weight:700}
.risk-high{padding:18px;border-radius:14px;background:#fef2f2;border:1px solid #fecaca;color:#991b1b;font-weight:700}
.customer-table{width:100%;border-collapse:collapse;background:white;font-size:12px}
.customer-table th{background:#1e3a8a;color:white;padding:8px;text-align:left}
.customer-table td{padding:7px;border-bottom:1px solid #e5e7eb}
</style>
<style>
/* ---------- 3D / Motion UI ---------- */
@keyframes floatCard {
  0%,100% { transform: translateY(0px) rotateX(0deg); }
  50% { transform: translateY(-5px) rotateX(1deg); }
}
@keyframes glowIn {
  from { opacity:0; transform:translateY(14px) scale(.985); }
  to { opacity:1; transform:translateY(0) scale(1); }
}
.hero {
  position: relative;
  overflow: hidden;
  transform-style: preserve-3d;
  animation: glowIn .65s ease-out;
}
.hero:before, .hero:after {
  content:"";
  position:absolute;
  border-radius:50%;
  filter:blur(2px);
  opacity:.18;
  pointer-events:none;
}
.hero:before {
  width:180px;height:180px;right:-30px;top:-70px;
  background:radial-gradient(circle, #93c5fd 0%, transparent 68%);
}
.hero:after {
  width:140px;height:140px;right:130px;bottom:-90px;
  background:radial-gradient(circle, #c4b5fd 0%, transparent 68%);
}
.kpi {
  transition: transform .25s ease, box-shadow .25s ease;
  transform-style: preserve-3d;
  animation: glowIn .5s ease-out;
}
.kpi:hover {
  transform: perspective(700px) translateY(-7px) rotateX(4deg) rotateY(-2deg);
  box-shadow: 0 14px 30px rgba(15,23,42,.14);
}
.card-3d {
  background:rgba(255,255,255,.86);
  border:1px solid rgba(148,163,184,.25);
  border-radius:18px;
  padding:20px;
  box-shadow:0 12px 35px rgba(15,23,42,.08);
  backdrop-filter:blur(8px);
  transform-style:preserve-3d;
  animation:glowIn .6s ease-out;
}
.neon-title {
  font-size:14px;
  font-weight:800;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:#2563eb;
}
.pill {
  display:inline-block;
  padding:7px 12px;
  border-radius:999px;
  background:#eff6ff;
  color:#1d4ed8;
  font-size:12px;
  font-weight:700;
  margin:4px 4px 0 0;
}
</style>

""", unsafe_allow_html=True)

DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL = "models/churn_model.pkl"
FEATURES = "models/feature_columns.pkl"

@st.cache_data
def load_data():
    x = pd.read_csv(DATA)
    x.columns = x.columns.str.strip()
    if "TotalCharges" in x:
        x["TotalCharges"] = pd.to_numeric(x["TotalCharges"], errors="coerce").fillna(0)
    return x

@st.cache_resource
def load_model():
    return joblib.load(MODEL), joblib.load(FEATURES)

try:
    df = load_data()
    model, feature_columns = load_model()
except Exception as e:
    st.error("Project files could not be loaded.")
    st.code(str(e))
    st.stop()

def predict_probability(values):
    row = pd.DataFrame([values])
    row = pd.get_dummies(row, drop_first=False)
    row = row.reindex(columns=list(feature_columns), fill_value=0)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(row)[0]
        return float(p[1])
    pred = model.predict(row)[0]
    return 1.0 if str(pred).lower() in ["yes","1","true"] else 0.0

st.sidebar.markdown("## 📊 Churn Analytics")
st.sidebar.caption("AI/ML Customer Retention Dashboard")
page = st.sidebar.radio("Dashboard Menu",
                        ["Overview","Churn Analysis","Customer Analytics","Churn Prediction"])

def render_3d_customer_map(dataframe):
    required = {"tenure", "MonthlyCharges", "TotalCharges", "Churn"}
    if not required.issubset(dataframe.columns):
        return
    sample = dataframe.sample(min(900, len(dataframe)), random_state=42).copy()
    sample["Churn Status"] = sample["Churn"].map({"No":"Retained", "Yes":"Churned"})
    fig3d = px.scatter_3d(
        sample,
        x="tenure",
        y="MonthlyCharges",
        z="TotalCharges",
        color="Churn Status",
        hover_data=["Contract", "InternetService"],
        title="3D Customer Risk Landscape",
        opacity=.72
    )
    fig3d.update_traces(marker=dict(size=4))
    fig3d.update_layout(
        height=560,
        margin=dict(l=0,r=0,t=55,b=0),
        scene=dict(
            xaxis_title="Tenure",
            yaxis_title="Monthly Charges",
            zaxis_title="Total Charges"
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Professional UI • Version 2.0")

st.markdown("""
<div class="hero">
<h1>Customer Churn Prediction & Analytics</h1>
<p>AI/ML powered dashboard for customer behavior, churn analysis and retention risk.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    total = len(df)
    churned = int((df.Churn=="Yes").sum())
    retained = int((df.Churn=="No").sum())
    rate = churned/total*100 if total else 0

    a,b,c,d = st.columns(4)
    for col,label,value,note in [
        (a,"👥 TOTAL CUSTOMERS",f"{total:,}","Customers in dataset"),
        (b,"⚠️ CHURNED",f"{churned:,}","Customers who left"),
        (c,"✅ RETAINED",f"{retained:,}","Customers retained"),
        (d,"📈 CHURN RATE",f"{rate:.2f}%","Overall churn")
    ]:
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div>{note}</div></div>', unsafe_allow_html=True)

    x,y = st.columns(2)
    with x:
        cd = df["Churn"].value_counts().reset_index()
        cd.columns=["Churn","Customers"]
        st.plotly_chart(px.pie(cd,names="Churn",values="Customers",hole=.55,title="Customer Churn Distribution"),use_container_width=True)
    with y:
        ct = pd.crosstab(df.Contract,df.Churn).reset_index().melt(id_vars="Contract",var_name="Churn",value_name="Customers")
        st.plotly_chart(px.bar(ct,x="Contract",y="Customers",color="Churn",barmode="group",title="Customers by Contract Type"),use_container_width=True)

elif page == "Churn Analysis":
    st.subheader("📈 Churn Analysis")
    x,y = st.columns(2)
    for col,title,target in [
        (x,"Churn by Contract Type","Contract"),
        (y,"Churn by Internet Service","InternetService")
    ]:
        data = pd.crosstab(df[target],df.Churn).reset_index().melt(id_vars=target,var_name="Churn",value_name="Customers")
        with col:
            st.plotly_chart(px.bar(data,x=target,y="Customers",color="Churn",barmode="group",title=title),use_container_width=True)
    data = pd.crosstab(df.PaymentMethod,df.Churn).reset_index().melt(id_vars="PaymentMethod",var_name="Churn",value_name="Customers")
    st.plotly_chart(px.bar(data,x="PaymentMethod",y="Customers",color="Churn",barmode="group",title="Churn by Payment Method"),use_container_width=True)

elif page == "Customer Analytics":
    st.subheader("👥 Customer Analytics")
    f1,f2,f3=st.columns(3)
    filtered=df.copy()
    with f1:
        vals=sorted(df.Contract.dropna().unique())
        selected=st.multiselect("Contract",vals,default=vals)
        filtered=filtered[filtered.Contract.isin(selected)]
    with f2:
        vals=sorted(df.InternetService.dropna().unique())
        selected=st.multiselect("Internet Service",vals,default=vals)
        filtered=filtered[filtered.InternetService.isin(selected)]
    with f3:
        selected=st.multiselect("Churn",["No","Yes"],default=["No","Yes"])
        filtered=filtered[filtered.Churn.isin(selected)]

    x,y=st.columns(2)
    with x:
        st.plotly_chart(px.histogram(filtered,x="tenure",color="Churn",nbins=30,title="Tenure Distribution"),use_container_width=True)
    with y:
        st.plotly_chart(px.histogram(filtered,x="MonthlyCharges",color="Churn",nbins=30,title="Monthly Charges Distribution"),use_container_width=True)

    st.markdown(f"**Customer Data — {len(filtered):,} records**")
    st.markdown('<style>.customer-table{width:100%;border-collapse:collapse}.customer-table th{background:#1e3a8a;color:white;padding:7px}.customer-table td{padding:6px;border-bottom:1px solid #ddd}</style>',unsafe_allow_html=True)
    st.markdown(filtered.head(100).to_html(index=False,classes="customer-table"),unsafe_allow_html=True)

else:
    st.subheader("🤖 Customer Churn Prediction")
    st.write("Enter customer information and estimate churn probability.")

    a,b,c=st.columns(3)
    with a:
        gender=st.selectbox("Gender",["Male","Female"])
        senior=st.selectbox("Senior Citizen",["No","Yes"])
        partner=st.selectbox("Partner",["No","Yes"])
        dependents=st.selectbox("Dependents",["No","Yes"])
        tenure=st.slider("Tenure (months)",0,72,12)
    with b:
        phone=st.selectbox("Phone Service",["No","Yes"])
        lines=st.selectbox("Multiple Lines",["No phone service","No","Yes"])
        internet=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        security=st.selectbox("Online Security",["No internet service","No","Yes"])
        backup=st.selectbox("Online Backup",["No internet service","No","Yes"])
    with c:
        device=st.selectbox("Device Protection",["No internet service","No","Yes"])
        support=st.selectbox("Tech Support",["No internet service","No","Yes"])
        contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
        paperless=st.selectbox("Paperless Billing",["No","Yes"])
        payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

    a,b=st.columns(2)
    with a: monthly=st.number_input("Monthly Charges",0.0,200.0,70.0,1.0)
    with b: total=st.number_input("Total Charges",0.0,10000.0,float(monthly*max(tenure,1)),10.0)
    tv=st.selectbox("Streaming TV",["No internet service","No","Yes"])
    movies=st.selectbox("Streaming Movies",["No internet service","No","Yes"])

    if st.button("🔮 Predict Churn",type="primary",use_container_width=True):
        values={"gender":gender,"SeniorCitizen":1 if senior=="Yes" else 0,"Partner":partner,"Dependents":dependents,
                "tenure":tenure,"PhoneService":phone,"MultipleLines":lines,"InternetService":internet,
                "OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,
                "StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,
                "PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}
        try:
            p=predict_probability(values)
            x,y=st.columns(2)
            with x: st.metric("Churn Probability",f"{p*100:.2f}%")
            with y:
                if p<.40:
                    st.markdown('<div class="risk-low">🟢 Customer is predicted to STAY<br><small>LOW RISK CUSTOMER</small></div>',unsafe_allow_html=True)
                elif p<.70:
                    st.markdown('<div class="risk-medium">🟠 Customer may be at risk<br><small>MEDIUM RISK CUSTOMER</small></div>',unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-high">🔴 Customer is likely to CHURN<br><small>HIGH RISK CUSTOMER</small></div>',unsafe_allow_html=True)
            st.progress(p)
            st.markdown(
                '<div class="card-3d"><span class="neon-title">Live AI Assessment</span><br>'
                '<span class="pill">Probability calculated</span>'
                '<span class="pill">Risk classified</span>'
                '<span class="pill">Retention action ready</span></div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

st.markdown('<p style="text-align:center;color:#64748b;padding:25px">Customer Churn Prediction & Analytics • Professional UI Version 2.0</p>',unsafe_allow_html=True)

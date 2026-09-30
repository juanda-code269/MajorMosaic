import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_predict,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

st.set_page_config(page_title="Major Outcomes Explorer",page_icon="📚",layout="wide")
st.title("Major Outcomes Explorer")
st.caption("Labor-market patterns are not a verdict on what anyone should study.")

PUBLIC_URL="https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv"


@st.cache_data(ttl=86400)
def public_data():
    d=pd.read_csv(PUBLIC_URL)
    return d.rename(columns={"Major":"major","Major_category":"category","Median":"median_earnings","Unemployment_rate":"unemployment_rate","Grad_total":"graduate_count","Total":"sample_size"})


@st.cache_data
def demo(n=170):
    rng=np.random.default_rng(14);cats=np.array(["STEM","Business","Humanities","Social Science","Health","Arts","Education"])
    category=rng.choice(cats,n);base={"STEM":72000,"Business":61000,"Humanities":45000,"Social Science":51000,"Health":65000,"Arts":42000,"Education":44000}
    earnings=np.array([base[c] for c in category])*rng.lognormal(0,.18,n)
    unemployment=np.clip(.095-earnings/1.5e6+rng.normal(0,.015,n),.015,.16)
    return pd.DataFrame({"major":[f"Demo field {i+1}" for i in range(n)],"category":category,"median_earnings":earnings,"unemployment_rate":unemployment,"sample_size":rng.integers(100,30000,n),"graduate_degree_share":rng.beta(2,4,n)})


source_choice=st.sidebar.radio("Data",["Public ACS-derived dataset","Synthetic offline demo","Upload CSV"])
try:
    if source_choice=="Public ACS-derived dataset":
        df=public_data();source="FiveThirtyEight college-majors data derived from the American Community Survey; historical, not current salary data."
    elif source_choice=="Upload CSV":
        up=st.sidebar.file_uploader("Major outcomes CSV",type="csv")
        if up is None:st.info("Upload a CSV to continue.");st.stop()
        df=pd.read_csv(up);source=f"Uploaded file: {up.name}"
    else:df=demo();source="Clearly labeled synthetic demonstration data"
except Exception as exc:
    st.warning(f"Public download unavailable ({exc}). Using synthetic demonstration data.");df=demo();source="Synthetic fallback after public-data failure"

required={"major","category","median_earnings","unemployment_rate"}
if not required.issubset(df.columns):st.error(f"Data needs columns: {', '.join(sorted(required))}");st.stop()
df=df.dropna(subset=list(required));numeric=df.select_dtypes(include=np.number).columns.tolist()
categories=sorted(df.category.astype(str).unique());selected=st.sidebar.multiselect("Fields",categories,default=categories)
filtered=df[df.category.astype(str).isin(selected)].copy()
inflation=st.sidebar.number_input("Illustrative inflation multiplier",.5,3.0,1.0,.01,help="Use 1.0 unless you have an externally justified conversion.")
filtered["adjusted_earnings"]=filtered.median_earnings*inflation
tabs=st.tabs(["Explore","Compare & rank","Regression","Methodology"])
with tabs[0]:
    st.info(source)
    c1,c2=st.columns(2)
    c1.plotly_chart(px.box(filtered,x="category",y="adjusted_earnings",points="outliers",title="Earnings distributions—not just averages"),width="stretch")
    size="sample_size" if "sample_size" in filtered else None
    c2.plotly_chart(px.scatter(filtered,x="unemployment_rate",y="adjusted_earnings",color="category",size=size,hover_name="major",title="Earnings and unemployment"),width="stretch")
with tabs[1]:
    criteria=st.selectbox("Sort by",["adjusted_earnings","unemployment_rate"]+[c for c in numeric if c not in {"median_earnings","unemployment_rate"}])
    ascending=criteria=="unemployment_rate"
    cols=[c for c in ["major","category","adjusted_earnings","unemployment_rate","sample_size","graduate_degree_share"] if c in filtered]
    st.dataframe(filtered.sort_values(criteria,ascending=ascending)[cols],width="stretch",hide_index=True)
    st.caption("This is a sortable comparison under your chosen criterion, not a definitive “best major” ranking.")
with tabs[2]:
    model_features=[c for c in ["category","unemployment_rate","sample_size","graduate_degree_share"] if c in df and c!="median_earnings"]
    X=df[model_features];y=df.median_earnings
    cat=[c for c in model_features if not pd.api.types.is_numeric_dtype(X[c])];num=[c for c in model_features if c not in cat]
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
    ridge=make_pipeline(prep,Ridge(alpha=10));pred=cross_val_predict(ridge,X,y,cv=KFold(5,shuffle=True,random_state=1))
    a,b=st.columns(2);a.metric("Cross-validated MAE",f"${mean_absolute_error(y,pred):,.0f}");b.metric("Cross-validated R²",f"{r2_score(y,pred):.2f}")
    st.plotly_chart(px.scatter(x=y,y=pred,labels={"x":"Observed median earnings","y":"Cross-validated estimate"},title="Association model diagnostics"),width="stretch")
    st.caption("The model summarizes associations among available fields. It cannot isolate the causal effect of choosing a major.")
with tabs[3]:
    st.markdown(f"""### Source and transformations
**Default public source:** [FiveThirtyEight college-majors repository]({PUBLIC_URL}), based on historical American Community Survey tabulations. Variables are renamed but not otherwise silently redefined. The inflation multiplier is user-supplied and defaults to 1.0; this app does not pretend it knows the correct comparison year.

Rows with missing required fields are removed. The displayed distribution preserves major-level variation; sample sizes are shown when present. The regression uses one-hot encoding, standardized numeric fields, ridge regularization, and shuffled five-fold cross-validation.

### Interpretation limits
These are observational, historical aggregates. Occupation, geography, degree level, industry, demographics, selection into majors, labor supply, working hours, and economic conditions confound comparisons. Median outcomes hide within-major dispersion, and inconsistent categories and sample sizes affect reliability. Correlation is not causation.""")


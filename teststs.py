# %%
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
import altair_viewer
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt

# %%
data = pd.DataFrame({'a': list('CCCDDDEEE'),
                     'b': [2, 7, 4, 1, 2, 6, 8, 4, 7]})
 


# %%
plot1 = alt.Chart(data).mark_point().encode(
    x='a',
    y='average(b)'
)
st.altair_chart(plot1)

plot2 = alt.Chart(data).mark_point().encode(
    y='a',
    x='average(b)'
)
st.altair_chart(plot2)
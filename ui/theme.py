# import streamlit as st


# def apply_theme():
#     """
#     Apply custom styling to the Streamlit application.
#     """

#     st.markdown(
#         """
#         <style>

#         /* ------------------------------
#            Main App
#         ------------------------------ */

#         .stApp{
#             background-color:#0E1117;
#             color:#F5F5F5;
#         }

#         .main .block-container{
#             padding-top:2rem;
#             padding-bottom:2rem;
#             padding-left:2rem;
#             padding-right:2rem;
#         }

#         /* ------------------------------
#            Sidebar
#         ------------------------------ */

#         section[data-testid="stSidebar"]{
#             background-color:#171B22;
#             border-right:1px solid #30363D;
#         }

#         /* ------------------------------
#            Metric Cards
#         ------------------------------ */

#         div[data-testid="metric-container"]{

#             background-color:#1B1F27;

#             border:1px solid #30363D;

#             padding:18px;

#             border-radius:12px;

#             transition:0.25s;
#         }

#         div[data-testid="metric-container"]:hover{

#             border:1px solid #F59E0B;

#             transform:translateY(-2px);
#         }

#         /* ------------------------------
#            Buttons
#         ------------------------------ */

#         .stButton>button{

#             width:100%;

#             border-radius:10px;

#             background-color:#F59E0B;

#             color:white;

#             border:none;

#             font-weight:600;
#         }

#         .stButton>button:hover{

#             background-color:#D97706;
#         }

#         /* ------------------------------
#            Tabs
#         ------------------------------ */

#         button[data-baseweb="tab"]{

#             font-size:16px;

#             font-weight:600;
#         }

#         /* ------------------------------
#            Expander
#         ------------------------------ */

#         .streamlit-expanderHeader{

#             font-size:16px;

#             font-weight:600;
#         }

#         /* ------------------------------
#            DataFrame
#         ------------------------------ */

#         .stDataFrame{

#             border-radius:10px;

#             overflow:hidden;
#         }

#         /* ------------------------------
#            Headings
#         ------------------------------ */

#         h1{

#             color:#F59E0B;
#         }

#         h2,h3{

#             color:#F5F5F5;
#         }

#         /* ------------------------------
#            Horizontal Line
#         ------------------------------ */

#         hr{

#             border-color:#30363D;
#         }

#         /* ------------------------------
#            Info Box
#         ------------------------------ */

#         div[data-testid="stAlert"]{

#             border-radius:10px;
#         }

#         </style>
#         """,
#         unsafe_allow_html=True
#     )


"""
theme.py

Centralized UI theme for the Streamlit dashboard.
"""

import streamlit as st

# ==========================================================
# Color Palette
# ==========================================================

PRIMARY = "#F59E0B"

SUCCESS = "#22C55E"

DANGER = "#EF4444"

WARNING = "#FACC15"

BACKGROUND = "#0E1117"

CARD = "#1B1F27"

BORDER = "#30363D"

TEXT = "#F5F5F5"

SUBTEXT = "#9CA3AF"

SIDEBAR = "#161B22"

# ==========================================================
# Theme
# ==========================================================


def apply_theme():
    """
    Applies the global Streamlit theme.
    """

    st.markdown(
        f"""
<style>

/* ==========================================================
Main App
========================================================== */

.stApp {{

    background-color: {BACKGROUND};

    color: {TEXT};

}}

.main .block-container {{

    max-width: 1500px;

    padding-top: 2rem;

    padding-left: 2rem;

    padding-right: 2rem;

    padding-bottom: 2rem;

}}

/* ==========================================================
Sidebar
========================================================== */

section[data-testid="stSidebar"] {{

    background-color: {SIDEBAR};

    border-right: 1px solid {BORDER};

}}

section[data-testid="stSidebar"] * {{

    color: {TEXT};

}}

/* ==========================================================
Buttons
========================================================== */

.stButton>button {{

    width:100%;

    border-radius:12px;

    border:none;

    padding:0.6rem;

    font-weight:600;

    background:{PRIMARY};

    color:white;

}}

.stButton>button:hover {{

    background:#D97706;

    transform:scale(1.02);

}}

/* ==========================================================
Metric Cards
========================================================== */

div[data-testid="metric-container"] {{

    background:{CARD};

    border:1px solid {BORDER};

    border-radius:14px;

    padding:18px;

}}

div[data-testid="metric-container"]:hover {{

    border:1px solid {PRIMARY};

}}

/* ==========================================================
Tabs
========================================================== */

button[data-baseweb="tab"] {{

    font-size:16px;

    font-weight:600;

}}

button[data-baseweb="tab"][aria-selected="true"] {{

    color:{PRIMARY};

}}

/* ==========================================================
Headers
========================================================== */

h1 {{

    color:{PRIMARY};

}}

h2,h3,h4 {{

    color:{TEXT};

}}

/* ==========================================================
Dataframes
========================================================== */

.stDataFrame {{

    border-radius:10px;

    overflow:hidden;

}}

/* ==========================================================
Expanders
========================================================== */

.streamlit-expanderHeader {{

    font-size:16px;

    font-weight:600;

}}

/* ==========================================================
Alerts
========================================================== */

div[data-testid="stAlert"] {{

    border-radius:12px;

}}

/* ==========================================================
Horizontal Rule
========================================================== */

hr {{

    border-color:{BORDER};

}}

/* ==========================================================
Scroll Bar
========================================================== */

::-webkit-scrollbar {{

    width:10px;

}}

::-webkit-scrollbar-thumb {{

    background:{PRIMARY};

    border-radius:20px;

}}

::-webkit-scrollbar-track {{

    background:{BACKGROUND};

}}

</style>
""",
        unsafe_allow_html=True,
    )
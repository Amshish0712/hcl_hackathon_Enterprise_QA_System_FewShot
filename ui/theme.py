"""Presentation helpers. Dynamic text is escaped before entering HTML."""

from html import escape

import streamlit as st


def apply_theme() -> None:
    st.markdown("""
        <style>
        .stApp { background: #081322; color: #e6edf7; }
        [data-testid="stSidebar"] { background: #0d1d33; border-right: 1px solid #203650; }
        .block-container { max-width: 1200px; padding-top: 2.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: -0.025em; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #243c59; border-radius: 12px;
        }
        [data-testid="stMetric"] { background: #11243c; padding: 1.2rem; border-radius: 12px; }
        .workspace-kicker { color: #8eabc9; font-size: .78rem; letter-spacing: .14em;
                            text-transform: uppercase; margin-bottom: .65rem; }
        .workspace-title { color: #f0f6ff; font-size: 2.4rem; font-weight: 700; line-height: 1.2; }
        .workspace-subtitle { color: #9db1ca; margin: .65rem 0 1.8rem; }
        .identity-name { color: #eef5ff; font-weight: 600; overflow-wrap: anywhere; }
        .identity-email { color: #9db1ca; font-size: .82rem; overflow-wrap: anywhere; }
        .role-badge { display: inline-block; margin-top: .65rem; background: #163f69;
                      color: #a9d5ff; padding: .2rem .7rem; border-radius: 999px;
                      font-size: .75rem; font-weight: 600; text-transform: uppercase; }
        </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="workspace-kicker">Secure Doc / Enterprise</div>'
        f'<div class="workspace-title">{escape(title)}</div>'
        f'<div class="workspace-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_identity(user: dict) -> None:
    st.markdown(
        f'<div class="identity-name">{escape(user["name"])}</div>'
        f'<div class="identity-email">{escape(user["email"])}</div>'
        f'<span class="role-badge">{escape(user["role"])}</span>',
        unsafe_allow_html=True,
    )

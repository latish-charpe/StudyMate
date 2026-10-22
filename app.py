import os
import re
import json
import shutil
from io import BytesIO
from pathlib import Path
import base64

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:
    FastEmbedEmbeddings = None

try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

from study_agent import StudyAgentTools, StudyAgentController

# ---------------- CONFIGURATION ----------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

API_KEY = GROQ_API_KEY if GROQ_API_KEY else GOOGLE_API_KEY
VECTOR_DIR = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)

CHAT_MODEL = GROQ_MODEL if GROQ_API_KEY else GEMINI_CHAT_MODEL
EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL

def get_asset_b64(filename: str) -> str:
    """Helper to convert local asset image to base64 for reliable HTML embedding."""
    for p in [Path(filename), Path("assets") / filename]:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""


st.set_page_config(
    page_title="StudyMate — Personal AI Study Space",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PURE LIGHT DESIGN SYSTEM (CSS) ----------------

STUDYMATE_LIGHT_THEME = """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&family=Caveat:wght@600;700&display=swap');

/* Typography & Global Font */
html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stTextInput, .stTextArea, .stSelectbox {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Base App Atmosphere */
.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Container Spacing */
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1420px !important;
}

/* ================= MUTED SLATE GRAY SIDEBAR ================= */
[data-testid="stSidebar"] {
    background-color: #e5e9f0 !important;
    background: #e5e9f0 !important;
    border-right: 1px solid #cbd5e1 !important;
}

/* Base text contrast in sidebar */
[data-testid="stSidebar"] {
    color: #0f172a !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.pdf-icon-badge),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
    font-weight: 600;
}

[data-testid="stSidebar"] hr {
    border-color: #cbd5e1 !important;
    margin: 14px 0 !important;
}

/* Sidebar Brand Header */
.sidebar-brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px 18px 4px;
}

.sidebar-brand-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    color: #ffffff;
}

.sidebar-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.sidebar-brand-tagline {
    font-size: 0.76rem;
    color: #475569 !important;
    font-weight: 500 !important;
    line-height: 1.35;
    margin-top: 2px;
}

/* Sidebar Radio Navigation Pill Menu */
[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {
    gap: 5px !important;
}

/* Hide all circular radio buttons, radio marks, dots and SVGs */
[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child,
[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:not(:last-child),
[data-testid="stSidebar"] div[data-testid="stRadio"] label > span:not(:last-child),
[data-testid="stSidebar"] div[data-testid="stRadio"] label input,
[data-testid="stSidebar"] div[data-testid="stRadio"] label svg,
[data-testid="stSidebar"] div[data-testid="stRadio"] [class*="RadioMark"],
[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:not(:last-child),
[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:not(:has([data-testid="stMarkdownContainer"])) {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

/* Clear, sharp text for menu items */
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {
    color: #0f172a !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:hover {
    background: rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:hover [data-testid="stMarkdownContainer"] p {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Active Selected Item - Blue Capsule Pill with Pure White Text */
[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Sidebar Section Title */
.sidebar-section-title,
.sidebar-section-collapse-header {
    font-size: 0.76rem;
    font-weight: 800;
    color: #475569 !important;
    margin: 14px 0 8px 0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Sidebar Upload Box Card - Elevated White Card */
.sidebar-upload-card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 16px 14px !important;
    text-align: center !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05) !important;
}

.sidebar-upload-icon {
    font-size: 1.8rem !important;
    margin-bottom: 4px !important;
    color: #2563eb !important;
    display: inline-block !important;
}

.sidebar-upload-head {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 2px !important;
}

.sidebar-upload-desc {
    font-size: 0.74rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}

/* STREAMLIT FILE UPLOADER IN SIDEBAR - ELEVATED WHITE DROPZONE */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: transparent !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1.5px dashed #94a3b8 !important;
    border-radius: 14px !important;
    padding: 14px 10px !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
    background-color: #f8fafc !important;
    border-color: #2563eb !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #0f172a !important;
    background: transparent !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 6px 16px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.45) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}

/* SIDEBAR MAIN BUTTON (Upload PDFs Button) - PURPLE GRADIENT PILL */
[data-testid="stSidebar"] div.stButton:not(:has([data-testid="stPopover"])) > button,
[data-testid="stSidebar"] button[key="btn_sidebar_upload"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border-radius: 9999px !important;
    padding: 9px 18px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4) !important;
    margin-top: 6px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] div.stButton:not(:has([data-testid="stPopover"])) > button:hover,
[data-testid="stSidebar"] button[key="btn_sidebar_upload"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.55) !important;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

[data-testid="stSidebar"] div.stButton:not(:has([data-testid="stPopover"])) > button * {
    color: #ffffff !important;
}

/* Sidebar Uploaded Files Section */
.sidebar-files-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 16px 0 8px 0;
}

.sidebar-files-title {
    font-size: 0.88rem;
    font-weight: 800;
    color: #0f172a !important;
}

.sidebar-count-badge {
    background: #cbd5e1;
    color: #0f172a;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 9999px;
    border: 1px solid #94a3b8;
}

.sidebar-file-item {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
}

.sidebar-file-item:hover {
    border-color: #94a3b8;
    background: #f8fafc;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
}

.sidebar-file-left {
    display: flex;
    align-items: center;
    gap: 10px;
    overflow: hidden;
}

.pdf-icon-badge {
    width: 28px;
    height: 28px;
    background: #ef4444;
    color: #ffffff;
    font-weight: 800;
    font-size: 0.62rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    flex-shrink: 0;
    letter-spacing: -0.02em;
}

.sidebar-file-meta {
    overflow: hidden;
}

.sidebar-file-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
}

.sidebar-file-size {
    font-size: 0.72rem;
    color: #475569 !important;
    font-weight: 500 !important;
}

.sidebar-file-dots {
    color: #64748b;
    font-size: 1.1rem;
    cursor: pointer;
    padding-left: 6px;
}

/* ================= TOP HEADER BAR ================= */
.top-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 18px;
}

.top-search-container {
    flex: 1;
    max-width: 540px;
    position: relative;
    display: flex;
    align-items: center;
}

.top-search-icon {
    position: absolute;
    left: 16px;
    color: #94a3b8;
    font-size: 0.95rem;
    pointer-events: none;
}

.top-search-input {
    width: 100%;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 9999px;
    padding: 10px 18px 10px 42px;
    font-size: 0.88rem;
    color: #334155;
    outline: none;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.02);
    transition: all 0.2s ease;
}

.top-search-input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.top-actions-right {
    display: flex;
    align-items: center;
    gap: 12px;
}

.top-icon-pill {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    color: #475569;
    cursor: pointer;
    position: relative;
    transition: all 0.2s ease;
}

.top-icon-pill:hover {
    background: #f8fafc;
    color: #0f172a;
}

.bell-red-badge {
    position: absolute;
    top: 7px;
    right: 8px;
    width: 7px;
    height: 7px;
    background: #ef4444;
    border-radius: 50%;
    border: 1.5px solid #ffffff;
}

.user-profile-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 9999px;
    padding: 4px 12px 4px 4px;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}

.user-avatar-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #4f46e5;
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.user-name-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #0f172a;
}

.user-down-arrow {
    font-size: 0.72rem;
    color: #64748b;
}

/* ================= HERO BANNER ================= */
.hero-banner-card {
    background-size: cover !important;
    background-position: center right !important;
    background-repeat: no-repeat !important;
    border: 1px solid #e2e8f0;
    border-radius: 28px;
    padding: 38px 46px;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 24px;
    align-items: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 22px;
    min-height: 290px;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.06);
}

.hero-left-content {
    z-index: 2;
    max-width: 520px;
}

.hero-pill-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    color: #4f46e5;
    border: 1px solid rgba(199, 210, 254, 0.8);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 14px;
    border-radius: 9999px;
    margin-bottom: 12px;
}

.hero-headline-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1.12;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin-bottom: 12px;
}

.hero-gradient-text {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle-text {
    color: #475569;
    font-size: 0.95rem;
    line-height: 1.55;
    max-width: 500px;
    margin-bottom: 0px;
}

/* Right Illustration Area of Hero: Floating Glass Pills & Handwritten Script */
.hero-right-visual {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    min-height: 230px;
    padding-left: 20px;
}

.hero-pills-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 2;
}

.hero-glass-pill {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 8px 16px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 0.84rem;
    font-weight: 700;
    color: #1e293b;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
    transition: transform 0.2s ease;
    width: fit-content;
}

.hero-glass-pill:hover {
    transform: translateX(4px);
    background: rgba(255, 255, 255, 0.96);
}

.hero-pill-icon {
    font-size: 1.05rem;
}

.hero-handwritten-note {
    position: absolute;
    right: 24px;
    top: 26px;
    font-family: 'Caveat', cursive;
    font-size: 1.65rem;
    font-weight: 700;
    color: #4338ca;
    line-height: 1.15;
    transform: rotate(-6deg);
    text-shadow: 0 2px 8px rgba(255, 255, 255, 0.9);
    z-index: 2;
}

/* Hero CTA Buttons */
button[key="btn_hero_start"] {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border-radius: 9999px !important;
    padding: 10px 22px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
}

button[key="btn_hero_explore"] {
    background: #ffffff !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 9999px !important;
    padding: 10px 22px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}

/* ================= 4 FEATURE CARDS ================= */
.feature-card-wrapper {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 195px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    transition: all 0.25s ease;
    margin-bottom: 8px;
}

.feature-card-wrapper:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    border-color: #cbd5e1;
}

.feature-card-top-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
}

.feature-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}

.feature-card-3d-thumb {
    width: 48px;
    height: 48px;
    object-fit: cover;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
}

.feature-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 5px;
}

.feature-card-desc {
    color: #64748b;
    font-size: 0.82rem;
    line-height: 1.48;
    margin-bottom: 0px;
    min-height: 48px;
}

/* Feature Card Pill Buttons */
button[key^="card_btn_"] {
    border-radius: 9999px !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

button[key="card_btn_notes"] {
    background: #eff6ff !important;
    color: #2563eb !important;
    border: 1px solid #bfdbfe !important;
}
button[key="card_btn_notes"]:hover {
    background: #2563eb !important;
    color: #ffffff !important;
}

button[key="card_btn_chat"] {
    background: #f5f3ff !important;
    color: #7c3aed !important;
    border: 1px solid #ddd6fe !important;
}
button[key="card_btn_chat"]:hover {
    background: #7c3aed !important;
    color: #ffffff !important;
}

button[key="card_btn_quiz"] {
    background: #fff7ed !important;
    color: #ea580c !important;
    border: 1px solid #fed7aa !important;
}
button[key="card_btn_quiz"]:hover {
    background: #ea580c !important;
    color: #ffffff !important;
}

button[key="card_btn_summary"] {
    background: #ecfdf5 !important;
    color: #059669 !important;
    border: 1px solid #a7f3d0 !important;
}
button[key="card_btn_summary"]:hover {
    background: #059669 !important;
    color: #ffffff !important;
}

/* ================= BOTTOM PANELS ================= */
.bottom-panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    margin-top: 10px;
    height: 100%;
}

.bottom-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.bottom-panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
}

.view-all-link {
    color: #2563eb !important;
    font-size: 0.84rem;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
}

/* Recently Uploaded Documents Rows */
.uploaded-doc-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}

.uploaded-doc-row:hover {
    background: #f1f5f9;
    border-color: #e2e8f0;
}

.doc-row-left {
    display: flex;
    align-items: center;
    gap: 12px;
    overflow: hidden;
}

.doc-row-meta {
    display: flex;
    flex-direction: column;
}

.doc-row-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: #0f172a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
}

.doc-row-sub {
    font-size: 0.74rem;
    color: #64748b;
}

.ready-pill-badge {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #059669;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
}

/* Quick Tips 3-Item Layout */
.quick-tips-container {
    display: flex;
    gap: 10px;
}

.quick-tip-card {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 14px;
    padding: 12px 14px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    transition: all 0.2s ease;
}

.quick-tip-card:hover {
    background: #ffffff;
    border-color: #e2e8f0;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.quick-tip-icon-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.88rem;
    font-weight: 700;
    flex-shrink: 0;
}

.quick-tip-body {
    overflow: hidden;
}

.quick-tip-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
    line-height: 1.25;
}

.quick-tip-desc {
    font-size: 0.72rem;
    color: #64748b;
    line-height: 1.35;
}

/* Hide Streamlit Header / Footer decoration */
header {visibility: hidden !important;}
footer {visibility: hidden !important;}
#MainMenu {visibility: hidden !important;}

/* ================= ASK YOUR NOTES PREMIUM DESIGN ================= */
.ask-breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 12px;
}

.ask-header-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px 26px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.ask-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.ask-book-icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #2563eb;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
}

.ask-header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 2px 0;
}

.ask-header-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

.ask-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #334155;
    font-size: 0.84rem;
    font-weight: 600;
    padding: 7px 16px;
    border-radius: 9999px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    white-space: nowrap;
}

.ask-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
}

/* Welcome Section Card */
.ask-welcome-card {
    background: linear-gradient(115deg, #f5f3ff 0%, #ede9fe 40%, #e0e7ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.04);
}

.ask-welcome-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}

.ask-welcome-desc {
    color: #475569;
    font-size: 0.94rem;
    line-height: 1.55;
    margin: 0;
    max-width: 680px;
}

/* 4 Suggestion Cards Grid */
button[key^="sugg_card_"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 20px 18px !important;
    text-align: left !important;
    min-height: 145px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

button[key^="sugg_card_"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.09) !important;
}

button[key="sugg_card_1"]:hover { border-color: #93c5fd !important; }
button[key="sugg_card_2"]:hover { border-color: #c4b5fd !important; }
button[key="sugg_card_3"]:hover { border-color: #fdba74 !important; }
button[key="sugg_card_4"]:hover { border-color: #86efac !important; }

button[key^="sugg_card_"] p {
    text-align: left !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    line-height: 1.4 !important;
}

button[key="sugg_card_1"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="sugg_card_2"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #f5f3ff;
    color: #7c3aed;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="sugg_card_3"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #fff7ed;
    color: #ea580c;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="sugg_card_4"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #ecfdf5;
    color: #059669;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

/* Start a Conversation Card */
.ask-start-convo-card {
    background: #ffffff;
    border: 1px dashed #cbd5e1;
    border-radius: 22px;
    padding: 38px 24px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
}

.ask-convo-bubble-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #f5f3ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin: 0 auto 12px auto;
}

.ask-convo-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
}

.ask-convo-desc {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.55;
    max-width: 540px;
    margin: 0 auto;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 18px 22px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02) !important;
    line-height: 1.65 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has(span:contains("👤")),
[data-testid="stChatMessage"]:has([aria-label="chat avatar 👤"]) {
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 18px !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
[data-testid="stChatMessage"]:has(span:contains("📖")),
[data-testid="stChatMessage"]:has([aria-label="chat avatar 📖"]) {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.03) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    margin-top: 14px !important;
    margin-bottom: 8px !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
    color: #334155 !important;
    padding-left: 20px !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    margin-bottom: 6px !important;
    line-height: 1.6 !important;
}

/* Source Box */
.source-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3.5px solid #4f46e5;
    border-radius: 12px;
    padding: 12px 16px;
    margin-top: 8px;
    transition: all 0.2s ease;
}

.source-box:hover {
    background: #ffffff;
    border-color: #c7d2fe;
    border-left-color: #4338ca;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.06);
}

.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
}

.source-excerpt {
    color: #475569;
    font-size: 0.86rem;
    line-height: 1.5;
}

/* Composer Input */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 9999px !important;
    padding: 12px 20px !important;
    font-size: 0.94rem !important;
    color: #0f172a !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4) !important;
}

/* ================= AI STUDY CHAT PREMIUM DESIGN ================= */
.aichat-icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #ddd6fe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #7c3aed;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.12);
}

.aichat-header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 2px 0;
}

.aichat-header-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

.aichat-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #334155;
    font-size: 0.84rem;
    font-weight: 600;
    padding: 7px 16px;
    border-radius: 9999px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    white-space: nowrap;
}

.aichat-welcome-card {
    background: linear-gradient(115deg, #f5f3ff 0%, #ede9fe 40%, #e0e7ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.04);
}

.aichat-welcome-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}

.aichat-welcome-desc {
    color: #475569;
    font-size: 0.94rem;
    line-height: 1.55;
    margin: 0;
    max-width: 720px;
}

/* 4 Suggestion Cards for AI Study Chat */
button[key^="aichat_card_"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 20px 18px !important;
    text-align: left !important;
    min-height: 145px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

button[key^="aichat_card_"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.09) !important;
}

button[key="aichat_card_1"]:hover { border-color: #c4b5fd !important; }
button[key="aichat_card_2"]:hover { border-color: #86efac !important; }
button[key="aichat_card_3"]:hover { border-color: #fdba74 !important; }
button[key="aichat_card_4"]:hover { border-color: #93c5fd !important; }

button[key^="aichat_card_"] p {
    text-align: left !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    line-height: 1.4 !important;
}

button[key="aichat_card_1"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #f5f3ff;
    color: #7c3aed;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="aichat_card_2"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #ecfdf5;
    color: #059669;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="aichat_card_3"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #fff7ed;
    color: #ea580c;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

button[key="aichat_card_4"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}


/* ================= PRACTICE QUIZ & SMART SUMMARIES PREMIUM STYLES ================= */
.quiz-header-icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1px solid #fecdd3;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #e11d48;
    box-shadow: 0 4px 12px rgba(225, 29, 72, 0.12);
}

.summary-header-icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #2563eb;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
}

.quiz-welcome-card {
    background: linear-gradient(115deg, #fff1f2 0%, #fdf2f8 40%, #f5f3ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(225, 29, 72, 0.04);
}

.summary-welcome-card {
    background: linear-gradient(115deg, #eff6ff 0%, #ede9fe 45%, #f5f3ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.04);
}

.config-card-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03);
}

/* 4 Suggestion Cards for Quiz & Summary */
button[key^="quiz_preset_"],
button[key^="summary_preset_"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 20px 18px !important;
    text-align: left !important;
    min-height: 145px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

button[key^="quiz_preset_"]:hover,
button[key^="summary_preset_"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.09) !important;
}

button[key^="quiz_preset_"] p,
button[key^="summary_preset_"] p {
    text-align: left !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    line-height: 1.4 !important;
}

button[key^="quiz_preset_"]::after,
button[key^="summary_preset_"]::after {
    content: "→";
    align-self: flex-end;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #f1f5f9;
    color: #4f46e5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92rem;
    margin-top: 8px;
}

/* Primary Button Styling across Main Area */
button[kind="primary"],
div.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border-radius: 9999px !important;
    padding: 10px 24px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover,
div.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

button[kind="primary"] *,
[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
}


/* ================= 3-DOTS POPOVER ACTION BUTTON ================= */
div[data-testid="stPopover"] {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

div[data-testid="stPopover"] > button,
button[data-testid="stPopoverButton"] {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    color: #64748b !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    padding: 0 !important;
    min-height: 28px !important;
    max-height: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    width: 28px !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
    overflow: hidden !important;
}

div[data-testid="stPopover"] > button:hover,
button[data-testid="stPopoverButton"]:hover {
    background: #cbd5e1 !important;
    background-color: #cbd5e1 !important;
    color: #0f172a !important;
    box-shadow: none !important;
    transform: none !important;
}

/* HIDE THE ARROW / CHEVRON COMPLETELY FROM POPOVER BUTTON */
div[data-testid="stPopover"] button [aria-hidden="true"],
div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
div[data-testid="stPopover"] button svg,
div[data-testid="stPopover"] button span:has([data-testid="stIconMaterial"]),
div[data-testid="stPopover"] button > div > *:not(:first-child),
div[data-testid="stPopover"] button > *:not([data-testid="stMarkdownContainer"]):not(:first-child),
button[data-testid="stPopoverButton"] [aria-hidden="true"],
button[data-testid="stPopoverButton"] [data-testid="stIconMaterial"],
button[data-testid="stPopoverButton"] svg,
button[data-testid="stPopoverButton"] > div > *:not(:first-child) {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Ensure the 3 dots label is perfectly centered */
div[data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
button[data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    line-height: 1 !important;
}

div[data-testid="stPopover"] button [data-testid="stMarkdownContainer"] p,
button[data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    line-height: 1 !important;
}

div[data-testid="stPopover"] button:hover [data-testid="stMarkdownContainer"] p,
button[data-testid="stPopoverButton"]:hover [data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
}

/* Popover dropdown body styling */
div[data-testid="stPopoverBody"] {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08) !important;
    padding: 10px 12px !important;
    min-width: 170px !important;
}

div[data-testid="stPopoverBody"] .stDownloadButton {
    margin-bottom: 6px !important;
}

div[data-testid="stPopoverBody"] .stDownloadButton button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    padding: 6px 10px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stPopoverBody"] .stDownloadButton button:hover {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border-color: #4f46e5 !important;
}

div[data-testid="stPopoverBody"] .stDownloadButton button:hover * {
    color: #ffffff !important;
}

div[data-testid="stPopoverBody"] div.stButton button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    background: #fef2f2 !important;
    color: #dc2626 !important;
    border: 1px solid #fecaca !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    padding: 6px 10px !important;
}

div[data-testid="stPopoverBody"] div.stButton button:hover {
    background: #dc2626 !important;
    color: #ffffff !important;
    border-color: #dc2626 !important;
}

div[data-testid="stPopoverBody"] div.stButton button:hover * {
    color: #ffffff !important;
}

/* Document Item Card Container styling */
div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge) {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge):hover {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge) [data-testid="stVerticalBlock"] {
    padding: 2px 4px !important;
}

/* Sidebar Document Container */
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge) {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    margin-bottom: 6px !important;
    padding: 0 !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge):hover {
    border-color: #94a3b8 !important;
    background: #f8fafc !important;
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(span.pdf-icon-badge) [data-testid="stVerticalBlock"] {
    padding: 4px 8px !important;
    gap: 0 !important;
}

/* AI Study Agent Workspace Styles */
.agent-header-icon-box {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
    color: #ffffff;
}

.agent-goal-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 20px;
}

.agent-roadmap-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 24px;
}

</style>
"""

st.markdown(STUDYMATE_LIGHT_THEME, unsafe_allow_html=True)


def get_hero_image_b64():
    """Load local hero 3D illustration asset as base64 string."""
    p = Path("assets/hero_books.jpg")
    if p.exists():
        try:
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""
    return ""

# ---------------- MODELS & CACHING ----------------

@st.cache_resource
def get_embeddings():
    """Initialize study assistant embeddings."""
    if FastEmbedEmbeddings is not None:
        try:
            return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            pass
    if GOOGLE_API_KEY and GoogleGenerativeAIEmbeddings is not None:
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return None


@st.cache_resource
def get_llm():
    """Initialize study assistant language model."""
    if GROQ_API_KEY and ChatGroq is not None:
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
        )
    elif GOOGLE_API_KEY and ChatGoogleGenerativeAI is not None:
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )
    return None


def clean_response(content):
    """
    Format AI responses into clean, readable text.
    Handles strings, LangChain message objects, and list/dictionary blocks.
    """
    if content is None:
        return ""

    if hasattr(content, "content"):
        content = content.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(clean_response(item["content"]))
                elif "parts" in item:
                    parts.append(clean_response(item["parts"]))
                else:
                    text_val = item.get("text") or item.get("content")
                    if text_val:
                        parts.append(str(text_val))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return clean_response(content["content"])
        if "parts" in content and isinstance(content["parts"], list):
            return clean_response(content["parts"])
        return str(content).strip()

    return str(content).strip()


def parse_quota_error(error):
    """
    Check if an error is due to usage limits (HTTP 429) and extract wait time.
    """
    msg = str(error)
    is_quota = (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "Quota exceeded" in msg
        or "rate_limit_exceeded" in msg
        or "rate limit" in msg.lower()
    )
    if not is_quota:
        return None

    retry_match = re.search(r"(?:retry in|try again in)\s+([0-9a-zA-Z\.\s]+?)(?:\.|$|,)", msg, re.IGNORECASE)
    delay_match = re.search(r"retryDelay[\'\":\s]+(\d+)s", msg)

    wait_time = None
    if retry_match:
        wait_time = retry_match.group(1).strip()
    elif delay_match:
        secs = int(delay_match.group(1))
        hours = secs // 3600
        mins = (secs % 3600) // 60
        secs_rem = secs % 60
        if hours > 0:
            wait_time = f"{hours}h {mins}m"
        elif mins > 0:
            wait_time = f"{mins}m {secs_rem}s"
        else:
            wait_time = f"{secs_rem}s"

    is_daily_limit = bool(
        "free_tier" in msg.lower()
        or "perday" in msg.lower()
        or "limit: 20" in msg
        or (wait_time and "h" in wait_time)
    )

    return {
        "is_quota": True,
        "wait_time": wait_time,
        "is_daily_limit": is_daily_limit,
    }


def render_student_error(error: Exception, action_label: str = "Request"):
    """Display user-friendly student error messages without internal stack traces."""
    quota_info = parse_quota_error(error)

    if quota_info:
        reset_str = f"in approximately **{quota_info['wait_time']}**" if quota_info["wait_time"] else "shortly once the quota window refreshes"
        st.warning(
            f"⏳ **Learning Assistant Limit Reached**\n\n"
            f"The rate limit or usage limit for your study assistant is currently exhausted.\n\n"
            f"Provider indicates the quota will reset {reset_str}.\n\n"
            f"**What you can do:**\n"
            f"- Please wait a moment before retrying.\n"
            f"- Or update the API key in your `.env` file if you have another active project key."
        )
    elif "API_KEY_INVALID" in str(error) or "invalid_api_key" in str(error).lower() or "400" in str(error) or "401" in str(error) or "authentication" in str(error).lower():
        st.error(
            "🔑 **API Key Not Recognized**\n\n"
            "The configured API key could not be verified. Please check your `.env` configuration file."
        )
    else:
        st.error(
            f"⚠️ **Could not complete {action_label}**\n\n"
            "An unexpected error occurred while preparing your answer. Please verify your connection or try again in a moment."
        )


def ask_gemini(prompt: str) -> str:
    """Invoke the language model and return clean student-facing text."""
    llm = get_llm()
    if not llm:
        raise ValueError("AI model is not initialized. Please configure GROQ_API_KEY (or GOOGLE_API_KEY) in .env.")
    response = llm.invoke(prompt)
    return clean_response(response)


# Convenient alias
ask_llm = ask_gemini


# ---------------- PDF PROCESSING ----------------

def extract_pdf(uploaded_file):
    """Extract readable text pages from an uploaded PDF file."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number,
                    },
                )
            )

    return documents


def create_vectorstore(uploaded_files):
    """Process study PDFs, split pages into study passages, and store locally."""
    documents = []

    for file in uploaded_files:
        documents.extend(extract_pdf(file))

    if not documents:
        raise ValueError(
            "No readable text was found in the uploaded documents. Please ensure your PDFs contain selectable text."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not extract study passages from the document text.")

    embeddings = get_embeddings()
    if not embeddings:
        raise ValueError("Embedding model could not be initialized.")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(VECTOR_DIR))

    return db, len(documents), len(chunks)


def load_vectorstore():
    """Safely load existing study index from disk."""
    index_file = VECTOR_DIR / "index.faiss"
    metadata_file = VECTOR_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        try:
            embeddings = get_embeddings()
            if not embeddings:
                return None
            return FAISS.load_local(
                str(VECTOR_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None

    return None


def get_indexed_sources(db):
    """Get unique document filenames currently indexed."""
    if db is None:
        return []
    try:
        sources = set()
        for doc in db.docstore._dict.values():
            src = doc.metadata.get("source")
            if src:
                sources.add(src)
        return sorted(list(sources))
    except Exception:
        return []


def get_all_chunks(db):
    """Retrieve all indexed passages from the docstore."""
    if db is None:
        return []
    try:
        return list(db.docstore._dict.values())
    except Exception:
        return []


def format_context(documents, max_chars_per_doc: int = 1500):
    """Format reference study passages with source citations."""
    parts = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Study Document")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        parts.append(
            f"[Source {i}: {source}, Page {page}]\n{content}"
        )

    return "\n\n".join(parts)


def retrieve(question: str, k: int = 4):
    """Retrieve relevant study passages from indexed materials."""
    db = st.session_state.get("vectorstore")
    if db is None:
        return []
    return db.similarity_search(question, k=k)



def get_document_download_data(target_filename: str):
    """Retrieve document bytes or extract notes content from vectorstore for download."""
    if "file_cache" in st.session_state and target_filename in st.session_state.file_cache:
        item = st.session_state.file_cache[target_filename]
        return item["bytes"], target_filename, item.get("mime", "application/pdf")

    # Extract text content from FAISS vectorstore docstore
    db = st.session_state.get("vectorstore")
    if db is not None:
        try:
            chunks = [
                doc for doc in db.docstore._dict.values()
                if doc.metadata.get("source") == target_filename
            ]
            if chunks:
                chunks.sort(key=lambda d: d.metadata.get("page", 0))
                content_parts = [f"# Study Notes: {target_filename}\n"]
                current_p = None
                for c in chunks:
                    p = c.metadata.get("page")
                    if p != current_p:
                        content_parts.append(f"\n--- Page {p} ---\n")
                        current_p = p
                    content_parts.append(c.page_content)
                text_data = "\n".join(content_parts)
                safe_name = target_filename
                if not safe_name.endswith(".txt") and not safe_name.endswith(".md"):
                    safe_name = safe_name.replace(".pdf", "") + "_notes.txt"
                return text_data.encode("utf-8"), safe_name, "text/plain"
        except Exception:
            pass

    # Fallback note for demo or indexed items
    demo_content = f"# Study Notes: {target_filename}\n\nContents for {target_filename} are indexed and ready for AI Study Chat, Practice Quizzes, and Smart Summaries."
    safe_name = target_filename.replace(".pdf", "") + "_notes.txt"
    return demo_content.encode("utf-8"), safe_name, "text/plain"


def delete_document(target_filename: str):
    """Safely remove a specific document from vectorstore or demo file list."""
    if "file_cache" in st.session_state and target_filename in st.session_state.file_cache:
        del st.session_state.file_cache[target_filename]

    # Remove from demo files list if present
    if "demo_files" in st.session_state:
        st.session_state.demo_files = [
            f for f in st.session_state.demo_files if f.get("name") != target_filename
        ]
    if "sidebar_demo_files" in st.session_state:
        st.session_state.sidebar_demo_files = [
            f for f in st.session_state.sidebar_demo_files if f.get("name") != target_filename
        ]

    db = st.session_state.get("vectorstore")
    if db is not None:
        try:
            all_docs = list(db.docstore._dict.values())
            remaining_docs = [
                d for d in all_docs if d.metadata.get("source") != target_filename
            ]
            if not remaining_docs:
                remove_study_materials()
            else:
                embeddings = get_embeddings()
                new_db = FAISS.from_documents(remaining_docs, embeddings)
                new_db.save_local(str(VECTOR_DIR))
                st.session_state.vectorstore = new_db
                st.session_state.messages = []
                st.session_state.quiz_data = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_user_answers = {}
                st.session_state.generated_summary = None
        except Exception:
            pass


def remove_study_materials():
    """Safely remove indexed documents from storage and state."""
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.quiz_data = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_user_answers = {}
    st.session_state.generated_summary = None
    if "file_cache" in st.session_state:
        st.session_state.file_cache.clear()

    try:
        if VECTOR_DIR.exists():
            for f in VECTOR_DIR.glob("*"):
                f.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------- SESSION STATE INITIALIZATION ----------------

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_vectorstore()
    except Exception:
        st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Quiz State
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []

if "quiz_current_index" not in st.session_state:
    st.session_state.quiz_current_index = 0

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# Summary State
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = None

if "summary_scope" not in st.session_state:
    st.session_state.summary_scope = None

# AI Study Agent State
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

if "agent_quiz_answers" not in st.session_state:
    st.session_state.agent_quiz_answers = {}

if "agent_input_goal" not in st.session_state:
    st.session_state.agent_input_goal = ""



if "demo_files" not in st.session_state:
    st.session_state.demo_files = [
        {"name": "BDE_ETL_vs_ELT.pdf", "size": "120 MB", "time": "Uploaded 2 hours ago"},
        {"name": "Machine_Learning.pdf", "size": "45 MB", "time": "Uploaded 1 day ago"},
    ]

if "sidebar_demo_files" not in st.session_state:
    st.session_state.sidebar_demo_files = [
        {"name": "BDE_ETL_vs_ELT.pdf", "size": "120 MB"},
        {"name": "Machine_Learning.pdf", "size": "45 MB"},
    ]

# ---------------- SIDEBAR NAVIGATION & DOCUMENT CONTROLS ----------------

if "ai_chat_messages" not in st.session_state:
    st.session_state.ai_chat_messages = []

if "ai_chat_query" not in st.session_state:
    st.session_state.ai_chat_query = None

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-wrapper">
            <div class="sidebar-brand-icon">🎓</div>
            <div>
                <div class="sidebar-brand-title">StudyMate</div>
                <div class="sidebar-brand-tagline">Your personal space<br>to learn smarter.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

        # 4 Unified Navigation Options
    NAV_OPTIONS = [
        "Home",
        "AI Study Chat",
        "Practice Quiz",
        "Smart Summaries",
    ]

    NAV_DISPLAY_MAP = {
        "Home": "🏠  Home",
        "AI Study Chat": "💬  AI Study Chat",
        "Practice Quiz": "🎯  Practice Quiz",
        "Smart Summaries": "📄  Smart Summaries",
    }
    REVERSE_NAV_MAP = {v: k for k, v in NAV_DISPLAY_MAP.items()}

    # Redirect legacy navigation selections if stored in session state
    if st.session_state.nav_page in ["Ask Your Notes", "AI Study Agent"]:
        st.session_state.nav_page = "AI Study Chat"

    current_display = NAV_DISPLAY_MAP.get(st.session_state.nav_page, "🏠  Home")
    selected_display = st.radio(
        "Navigation",
        list(NAV_DISPLAY_MAP.values()),
        index=list(NAV_DISPLAY_MAP.values()).index(current_display),
        label_visibility="collapsed",
    )
    selected_page = REVERSE_NAV_MAP.get(selected_display, "Home")
    if selected_page != st.session_state.nav_page:
        st.session_state.nav_page = selected_page
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # STUDY DOCUMENTS Section Title
    st.markdown("<div class='sidebar-section-title'>STUDY DOCUMENTS</div>", unsafe_allow_html=True)

    # Upload Notes Card
    st.markdown(
        """
        <div class="sidebar-upload-card">
            <div class="sidebar-upload-icon">☁️</div>
            <div class="sidebar-upload-head">Upload Notes</div>
            <div class="sidebar-upload-desc">PDF files (Max 200MB)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_uploads = st.file_uploader(
        "Upload PDF notes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Upload PDFs", key="btn_sidebar_upload", use_container_width=True):
        if not sidebar_uploads:
            st.warning("Please choose at least one PDF file.")
        else:
            try:
                with st.spinner("Preparing your study material…"):
                    if "file_cache" not in st.session_state:
                        st.session_state.file_cache = {}
                    for up_f in sidebar_uploads:
                        try:
                            st.session_state.file_cache[up_f.name] = {
                                "bytes": up_f.getvalue(),
                                "mime": "application/pdf",
                            }
                        except Exception:
                            pass
                    db, pages, chunks = create_vectorstore(sidebar_uploads)
                    st.session_state.vectorstore = db
                    st.session_state.messages = []
                    st.session_state.quiz_data = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_user_answers = {}
                    st.session_state.quiz_current_index = 0
                    st.session_state.generated_summary = None
                st.success("Study material prepared successfully!")
                st.rerun()
            except Exception as error:
                render_student_error(error, "Document Preparation")

    # Uploaded Files Section
    active_sources = get_indexed_sources(st.session_state.get("vectorstore"))
    sidebar_demo_files = [
        {"name": s, "size": "Ready"} for s in active_sources
    ] if active_sources else st.session_state.get("sidebar_demo_files", [])

    file_count = len(sidebar_demo_files)

    st.markdown(
        f"""
        <div class="sidebar-files-header">
            <span class="sidebar-files-title">Uploaded Files</span>
            <span class="sidebar-count-badge">{file_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, f in enumerate(sidebar_demo_files):
        with st.container(border=True):
            sb_info, sb_menu = st.columns([4.2, 0.8], vertical_alignment="center")
            with sb_info:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; overflow:hidden;">
                        <span class="pdf-icon-badge">PDF</span>
                        <div style="display:flex; flex-direction:column; overflow:hidden; line-height:1.2;">
                            <span class="sidebar-file-name" title="{f['name']}" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:700; font-size:0.83rem; color:#0f172a;">{f['name']}</span>
                            <span class="sidebar-file-size" style="font-size:0.72rem; color:#64748b; font-weight:500;">{f['size']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sb_menu:
                with st.popover("⋮"):
                    st.caption(f"**{f['name']}**")
                    dl_bytes, dl_name, dl_mime = get_document_download_data(f['name'])
                    st.download_button(
                        label="📥 Download Notes",
                        data=dl_bytes,
                        file_name=dl_name,
                        mime=dl_mime,
                        key=f"btn_sb_dl_{idx}_{abs(hash(f['name']))}",
                        use_container_width=True,
                    )
                    if st.button("🗑️ Delete", key=f"btn_sb_del_{idx}_{abs(hash(f['name']))}", use_container_width=True):
                        delete_document(f['name'])
                        st.rerun()

    if active_sources:
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Remove Uploaded Notes", use_container_width=True):
            remove_study_materials()
            st.rerun()


# ---------------- MAIN APP SCREEN ROUTING ----------------

# Missing API Key Warning
if not API_KEY:
    st.warning(
        "⚠️ **Configuration Needed**: Your API key is missing. "
        "Please make sure `GROQ_API_KEY` (or `GOOGLE_API_KEY`) is present in your `.env` file."
    )
    st.stop()


# =========================================================
# 1. HOME DASHBOARD
# =========================================================
if st.session_state.nav_page == "Home":
    # Top Search Bar & Profile Bar (with Sun Toggle & User Latish)
    st.markdown(
        """
        <div class="top-header-row">
            <div class="top-search-container">
                <span class="top-search-icon">🔍</span>
                <input type="text" class="top-search-input" placeholder="Search your notes, ask anything..." />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner (Using user's provided bg.png)
    bg_b64 = get_asset_b64("bg.png")
    bg_style = f"background-image: url('data:image/png;base64,{bg_b64}');" if bg_b64 else "background: linear-gradient(115deg, #f5f3ff 0%, #ede9fe 40%, #e0e7ff 75%, #eff6ff 100%);"

    st.markdown(
        f"""
        <div class="hero-banner-card" style="{bg_style}">
            <div class="hero-left-content">
                <div class="hero-pill-badge">YOUR PERSONAL LEARNING SPACE</div>
                <div class="hero-headline-big">
                    Learn smarter.<br>
                    <span class="hero-gradient-text">Remember more.</span>
                </div>
                <p class="hero-subtitle-text">
                    Turn your study material into clear answers, interactive practice, and revision-ready notes.
                </p>
            </div>
            <div class="hero-right-visual">
                <div class="hero-pills-stack">
                    <div class="hero-glass-pill">
                        <span class="hero-pill-icon">🎓</span> <span>Understand faster</span>
                    </div>
                    <div class="hero-glass-pill">
                        <span class="hero-pill-icon">📊</span> <span>Practice effectively</span>
                    </div>
                    <div class="hero-glass-pill">
                        <span class="hero-pill-icon">📄</span> <span>Prepare better</span>
                    </div>
                </div>
                <div class="hero-handwritten-note">
                    Same Notes,<br>Better Results ⤷
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Action Buttons
    col_h_b1, col_h_b2, col_h_b3, col_h_spacer = st.columns([1.8, 1.8, 1.8, 1.6])
    with col_h_b1:
        if st.button("💬 Start AI Study Chat →", key="btn_hero_chat", use_container_width=True, type="primary"):
            st.session_state.nav_page = "AI Study Chat"
            st.rerun()
    with col_h_b2:
        if st.button("🎯 Practice Quiz →", key="btn_hero_quiz", use_container_width=True):
            st.session_state.nav_page = "Practice Quiz"
            st.rerun()
    with col_h_b3:
        if st.button("📄 Smart Summaries →", key="btn_hero_summary", use_container_width=True):
            st.session_state.nav_page = "Smart Summaries"
            st.rerun()

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3 Primary Feature Cards (With 3D Icons & Specific Copy)
    chat_b64 = get_asset_b64("card_notes.jpg") or get_asset_b64("card_robot.jpg")
    quiz_b64 = get_asset_b64("card_quiz.jpg")
    summary_b64 = get_asset_b64("card_summary.jpg")

    col1, col2, col3 = st.columns(3)

    with col1:
        img_thumb = f'<img src="data:image/jpeg;base64,{chat_b64}" class="feature-card-3d-thumb" alt="chat" />' if chat_b64 else '<span>💬</span>'
        st.markdown(
            f"""
            <div class="feature-card-wrapper">
                <div>
                    <div class="feature-card-top-row">
                        <div class="feature-icon-box" style="background:#eff6ff; color:#2563eb;">💬</div>
                        {img_thumb}
                    </div>
                    <div class="feature-card-title">AI Study Chat</div>
                    <div class="feature-card-desc">
                        Ask any academic doubt or query your uploaded study notes with instant explanations and citations.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Chatting →", key="card_btn_chat", use_container_width=True):
            st.session_state.nav_page = "AI Study Chat"
            st.rerun()

    with col2:
        img_thumb = f'<img src="data:image/jpeg;base64,{quiz_b64}" class="feature-card-3d-thumb" alt="quiz" />' if quiz_b64 else '<span>📝</span>'
        st.markdown(
            f"""
            <div class="feature-card-wrapper">
                <div>
                    <div class="feature-card-top-row">
                        <div class="feature-icon-box" style="background:#fff7ed; color:#ea580c;">🎯</div>
                        {img_thumb}
                    </div>
                    <div class="feature-card-title">Practice Quiz</div>
                    <div class="feature-card-desc">
                        Test your understanding with custom MCQs, instant grading, and clear explanations.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate Quiz →", key="card_btn_quiz", use_container_width=True):
            st.session_state.nav_page = "Practice Quiz"
            st.rerun()

    with col3:
        img_thumb = f'<img src="data:image/jpeg;base64,{summary_b64}" class="feature-card-3d-thumb" alt="summary" />' if summary_b64 else '<span>📁</span>'
        st.markdown(
            f"""
            <div class="feature-card-wrapper">
                <div>
                    <div class="feature-card-top-row">
                        <div class="feature-icon-box" style="background:#ecfdf5; color:#059669;">📄</div>
                        {img_thumb}
                    </div>
                    <div class="feature-card-title">Smart Summaries</div>
                    <div class="feature-card-desc">
                        Turn lengthy notes into concise, well-structured revision guides with key concepts.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Create Summary →", key="card_btn_summary", use_container_width=True):
            st.session_state.nav_page = "Smart Summaries"
            st.rerun()

    st.markdown("<div style=\'margin-top: 14px;\'></div>", unsafe_allow_html=True)

    # Bottom Panels: Recently Uploaded (Left) + Quick Tips (Right)
    col_recent, col_tips = st.columns([1.05, 1.25])

    with col_recent:
        docs_to_show = [
            {"name": s, "size": "Active", "time": "Ready to study"}
            for s in active_sources
        ] if active_sources else st.session_state.get("demo_files", [])

        st.markdown(
            f"""
            <div class="bottom-panel-header" style="margin-bottom: 14px;">
                <div class="bottom-panel-title">
                    <span>📁</span> <span>Recently Uploaded</span>
                </div>
                <span style="font-size:0.8rem; font-weight:700; color:#4f46e5;">{len(docs_to_show)} Documents</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not docs_to_show:
            st.markdown(
                """
                <div style="background:#f8fafc; border:1.5px dashed #cbd5e1; border-radius:14px; padding:28px 16px; text-align:center; color:#64748b; font-size:0.88rem;">
                    No uploaded documents. Upload PDF files in the sidebar.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for idx, d in enumerate(docs_to_show):
                with st.container(border=True):
                    r_col_info, r_col_badge, r_col_pop = st.columns([4.2, 1.3, 0.5], vertical_alignment="center")
                    with r_col_info:
                        st.markdown(
                            f"""
                            <div class="doc-row-left">
                                <span class="pdf-icon-badge">PDF</span>
                                <div class="doc-row-meta">
                                    <span class="doc-row-title" title="{d['name']}">{d['name']}</span>
                                    <span class="doc-row-sub">{d['size']} • {d['time']}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with r_col_badge:
                        st.markdown(
                            """<div style="display:flex; justify-content:center;"><span class="ready-pill-badge"><span class="status-dot"></span> Ready</span></div>""",
                            unsafe_allow_html=True,
                        )
                    with r_col_pop:
                        with st.popover("⋮"):
                            st.markdown(f"<div style='font-size:0.88rem; font-weight:700; color:#0f172a; margin-bottom:4px;'>{d['name']}</div>", unsafe_allow_html=True)
                            st.caption("Document actions")
                            dl_bytes, dl_name, dl_mime = get_document_download_data(d['name'])
                            st.download_button(
                                label="📥 Download Notes",
                                data=dl_bytes,
                                file_name=dl_name,
                                mime=dl_mime,
                                key=f"btn_recent_dl_{idx}_{abs(hash(d['name']))}",
                                use_container_width=True,
                            )
                            if st.button("🗑️ Delete File", key=f"btn_del_recent_{idx}_{abs(hash(d['name']))}", use_container_width=True):
                                delete_document(d['name'])
                                st.rerun()

    with col_tips:
        st.markdown(
            """
            <div class="bottom-panel-card">
                <div class="bottom-panel-header">
                    <div class="bottom-panel-title">
                        <span>💡</span> <span>Quick Tips</span>
                    </div>
                </div>
                <div class="quick-tips-container">
                    <div class="quick-tip-card">
                        <div class="quick-tip-icon-circle" style="background:#eff6ff; color:#2563eb;">?</div>
                        <div class="quick-tip-body">
                            <div class="quick-tip-title">Ask specific questions</div>
                            <div class="quick-tip-desc">Get more accurate answers from your notes.</div>
                        </div>
                    </div>
                    <div class="quick-tip-card">
                        <div class="quick-tip-icon-circle" style="background:#f5f3ff; color:#7c3aed;">🎯</div>
                        <div class="quick-tip-body">
                            <div class="quick-tip-title">Try topic-based quizzes</div>
                            <div class="quick-tip-desc">Practice important concepts regularly.</div>
                        </div>
                    </div>
                    <div class="quick-tip-card">
                        <div class="quick-tip-icon-circle" style="background:#eff6ff; color:#4f46e5;">📄</div>
                        <div class="quick-tip-body">
                            <div class="quick-tip-title">Generate summaries</div>
                            <div class="quick-tip-desc">Create quick revision notes before exams.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 2. AI STUDY CHAT (UNIFIED PERSONAL TUTOR & NOTES INTELLIGENCE)
# =========================================================
elif st.session_state.nav_page == "AI Study Chat":
    active_sources = get_indexed_sources(st.session_state.get("vectorstore"))
    doc_count = len(active_sources) if active_sources else 0

    # Top Search Bar
    st.markdown(
        """
        <div class="top-header-row">
            <div class="top-search-container">
                <span class="top-search-icon">🔍</span>
                <input type="text" class="top-search-input" placeholder="Search topics, ask anything from your notes or general subjects..." />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breadcrumb: Home / AI Study Chat
    st.markdown(
        """
        <div class="ask-breadcrumb">
            <span style="color:#6366f1;">🏠</span>
            <span style="color:#64748b;">Home</span>
            <span style="color:#94a3b8;">/</span>
            <span style="color:#0f172a; font-weight:700;">AI Study Chat</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Header Card with Dynamic Status
    doc_pill_label = f"📄 {doc_count} Note{'s' if doc_count != 1 else ''} Attached" if doc_count else "⚡ 24/7 AI Tutor"
    status_label = "Notes Search Active" if doc_count else "Online"

    st.markdown(
        f"""
        <div class="ask-header-card">
            <div class="ask-header-left">
                <div class="aichat-icon-box">💬</div>
                <div>
                    <h1 class="aichat-header-title">AI Study Chat</h1>
                    <p class="aichat-header-subtitle">Your 24/7 personal tutor. Answers questions grounded in your uploaded notes or general academic knowledge.</p>
                </div>
            </div>
            <div>
                <div class="ask-status-pill">
                    <span>{doc_pill_label}</span>
                    <span style="color:#cbd5e1;">•</span>
                    <span style="display:inline-flex; align-items:center; gap:5px; color:#059669; font-weight:700;">
                        <span class="ask-status-dot"></span> {status_label}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Welcome Banner Card
    st.markdown(
        """
        <div class="aichat-welcome-card">
            <div class="aichat-welcome-head">What would you like to master today?</div>
            <p class="aichat-welcome-desc">
                Ask questions across your uploaded study materials, math, computer science, exam prep, or difficult concepts. Get clear explanations, step-by-step breakdowns, code/math examples, and verified source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty State: Suggestion Cards & Quick Starter Prompts
    if not st.session_state.ai_chat_messages:
        st.markdown(
            """
            <div class="ask-start-convo-card">
                <div class="ask-convo-bubble-icon" style="background:#f5f3ff; color:#7c3aed;">💡</div>
                <div class="ask-convo-head">Start an AI tutoring session</div>
                <p class="ask-convo-desc">
                    Type any academic doubt, question about your notes, or concept below.<br>
                    I'll provide clear, step-by-step answers with citations whenever your notes cover the topic.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Starter Chips
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if doc_count > 0:
                p1 = "What are the core concepts and key formulas in my notes?"
                p2 = "Give me a quick 5-bullet summary of the main points."
            else:
                p1 = "Explain the difference between supervised and unsupervised learning."
                p2 = "How do I build an effective study revision schedule?"
            if st.button(f"💡 {p1}", key="btn_chip_1", use_container_width=True):
                st.session_state.ai_chat_query = p1
                st.rerun()
            if st.button(f"🔍 {p2}", key="btn_chip_2", use_container_width=True):
                st.session_state.ai_chat_query = p2
                st.rerun()

        with col_q2:
            if doc_count > 0:
                p3 = "What are the most likely exam questions from this document?"
                p4 = "Explain the hardest topic from my notes in simple terms with an example."
            else:
                p3 = "What are high-yield study techniques for rapid retention?"
                p4 = "Break down a complex technical concept step-by-step."
            if st.button(f"❓ {p3}", key="btn_chip_3", use_container_width=True):
                st.session_state.ai_chat_query = p3
                st.rerun()
            if st.button(f"🧠 {p4}", key="btn_chip_4", use_container_width=True):
                st.session_state.ai_chat_query = p4
                st.rerun()
    else:
        # Clear Chat Button Row if messages exist
        col_chat_stat, col_chat_clr = st.columns([3.2, 0.8])
        with col_chat_clr:
            if st.button("🧹 Clear Chat", key="btn_clear_aichat_tab", use_container_width=True):
                st.session_state.ai_chat_messages = []
                st.session_state.ai_chat_query = None
                st.session_state.messages = []
                st.rerun()

        # Conversation History
        for msg in st.session_state.ai_chat_messages:
            role = msg["role"]
            avatar = "🧑‍🎓" if role == "user" else "💬"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("sources"):
                    with st.expander(f"📚 Verified Sources ({len(msg['sources'])} Citations from Notes)", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="source-tag">📄 {src.get('source', 'Document')} · Page {src.get('page', '?')}</div>
                                    <div class="source-excerpt">"{src.get('snippet', '')}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat Composer Input
    ai_query = st.chat_input("Ask any question on your notes, concepts, or homework…")

    active_ai_query = None
    if st.session_state.get("ai_chat_query"):
        active_ai_query = st.session_state.ai_chat_query
        st.session_state.ai_chat_query = None
    elif ai_query and ai_query.strip():
        active_ai_query = ai_query.strip()

    if active_ai_query:
        clean_q = active_ai_query.strip()
        st.session_state.ai_chat_messages.append({"role": "user", "content": clean_q})

        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(clean_q)

        with st.chat_message("assistant", avatar="💬"):
            try:
                # Check if notes are indexed and retrieve relevant passages
                docs = retrieve(clean_q, k=4) if doc_count > 0 else []

                if docs:
                    with st.spinner("Searching your study notes & preparing answer…"):
                        context = format_context(docs)
                        prompt = f"""You are StudyMate's dedicated educational AI assistant and tutor.

Answer the student's question clearly, pedagogically, and accurately.

RELEVANT EXCERPTS FROM UPLOADED NOTES:
{context}

STUDENT QUESTION:
{clean_q}

INSTRUCTIONS:
1. Base your answer primarily on the uploaded study context when relevant, explaining concepts clearly and highlighting key points.
2. If the notes only cover part of the answer or if the student asks for broader clarification/examples, provide a complete academic explanation while citing details that come from their notes.
3. Structure your response cleanly with headings, bullet points, and code/math blocks where helpful.

Provide a student-friendly, thorough answer:"""
                        answer = ask_gemini(prompt)

                        sources_meta = []
                        seen_sources = set()
                        for d in docs:
                            s_name = d.metadata.get("source", "Study Document")
                            s_page = d.metadata.get("page", "?")
                            key = (s_name, s_page)
                            if key not in seen_sources:
                                seen_sources.add(key)
                                snippet = " ".join(d.page_content.strip()[:180].split())
                                sources_meta.append({
                                    "source": s_name,
                                    "page": s_page,
                                    "snippet": snippet,
                                })

                        st.markdown(answer)

                        if sources_meta:
                            with st.expander(f"📚 Verified Sources ({len(sources_meta)} Citations)", expanded=False):
                                for src in sources_meta:
                                    st.markdown(
                                        f"""
                                        <div class="source-box">
                                            <div class="source-tag">📄 {src['source']} · Page {src['page']}</div>
                                            <div class="source-excerpt">"{src['snippet']}"</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.ai_chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_meta,
                        })
                else:
                    with st.spinner("Thinking & preparing academic explanation…"):
                        tutoring_prompt = f"""You are StudyMate's dedicated educational AI tutor.
Provide a clear, engaging, pedagogically structured, and easy-to-understand explanation for the student's question.
Use clear formatting, headings, bullet points, and code/math blocks where helpful.

QUESTION:
{clean_q}

Provide a comprehensive, student-friendly answer:"""
                        resp = ask_gemini(tutoring_prompt)
                        st.markdown(resp)
                        st.session_state.ai_chat_messages.append({
                            "role": "assistant",
                            "content": resp,
                            "sources": [],
                        })
            except Exception as error:
                render_student_error(error, "AI Study Chat")



# =========================================================
# 3. PRACTICE QUIZ (INTERACTIVE ASSESSMENTS)
# =========================================================
elif st.session_state.nav_page == "Practice Quiz":
    active_sources = get_indexed_sources(st.session_state.get("vectorstore"))
    doc_count = len(active_sources) if active_sources else 0

    # Top Search Bar
    st.markdown(
        """
        <div class="top-header-row">
            <div class="top-search-container">
                <span class="top-search-icon">🔍</span>
                <input type="text" class="top-search-input" placeholder="Search quiz topics, test yourself..." />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breadcrumb: Home / Practice Quiz
    st.markdown(
        """
        <div class="ask-breadcrumb">
            <span style="color:#6366f1;">🏠</span>
            <span style="color:#64748b;">Home</span>
            <span style="color:#94a3b8;">/</span>
            <span style="color:#0f172a; font-weight:700;">Practice Quiz</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Header Card: Target Icon, Title, Subtitle & Ready Pill
    doc_pill_label = f"📄 {doc_count} Note{'s' if doc_count != 1 else ''} Attached" if doc_count else "⚡ Quiz Ready"
    st.markdown(
        f"""
        <div class="ask-header-card">
            <div class="ask-header-left">
                <div class="quiz-header-icon-box">🎯</div>
                <div>
                    <h1 class="aichat-header-title">Practice Quiz</h1>
                    <p class="aichat-header-subtitle">Test your knowledge with multiple-choice questions crafted from your study notes.</p>
                </div>
            </div>
            <div>
                <div class="ask-status-pill">
                    <span>{doc_pill_label}</span>
                    <span style="color:#cbd5e1;">•</span>
                    <span style="display:inline-flex; align-items:center; gap:5px; color:#059669; font-weight:700;">
                        <span class="ask-status-dot"></span> Ready
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    if st.session_state.vectorstore is None:
        st.markdown(
            """
            <div class="ask-start-convo-card">
                <div class="ask-convo-bubble-icon" style="background:#fff1f2; color:#e11d48;">📚</div>
                <div class="ask-convo-head">Study notes required for quiz</div>
                <p class="ask-convo-desc">
                    Please upload your course PDFs in the left sidebar to generate personalized practice quizzes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Welcome Banner Card
        st.markdown(
            """
            <div class="quiz-welcome-card">
                <div class="aichat-welcome-head">Generate personalized quizzes from your notes</div>
                <p class="aichat-welcome-desc">
                    Challenge yourself with interactive multiple-choice questions grounded directly in your study materials. Review your score and learn from detailed explanations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Elevated Quiz Configuration Box
        with st.expander("⚙️ Configure Practice Quiz Options", expanded=len(st.session_state.quiz_data) == 0):
            col_t, col_c = st.columns([3, 1])
            with col_t:
                default_topic = st.session_state.get("quiz_preset_topic", "")
                quiz_topic = st.text_input(
                    "Specific Topic (optional)",
                    value=default_topic,
                    placeholder="e.g. ETL vs ELT, Neural Networks, Probability Distributions",
                    help="Leave blank to cover concepts across all uploaded pages.",
                )
            with col_c:
                preset_c = st.session_state.get("quiz_preset_count", 5)
                c_idx = [5, 10, 15].index(preset_c) if preset_c in [5, 10, 15] else 0
                quiz_count = st.selectbox("Number of Questions", [5, 10, 15], index=c_idx)

            generate_quiz_btn = st.button("🚀 Create Practice Quiz", type="primary", use_container_width=True)

        # Trigger generation either from button or preset card
        should_generate = generate_quiz_btn or st.session_state.get("trigger_quiz_gen", False)
        if st.session_state.get("trigger_quiz_gen"):
            st.session_state.trigger_quiz_gen = False

        if should_generate:
            search_query = quiz_topic.strip() if quiz_topic.strip() else "key concepts, definitions, and main principles"
            try:
                with st.spinner("Generating your personalized practice quiz…"):
                    docs = retrieve(search_query, k=8)
                    if not docs:
                        st.warning("No relevant study content found for this topic.")
                    else:
                        context = format_context(docs)

                        prompt = f"""Create exactly {quiz_count} multiple-choice practice questions from the supplied study material.

RULES:
- Base questions ONLY on the facts present in the study material.
- Each question must have four distinct options labeled exactly: A, B, C, D.
- Include exactly one correct answer (the letter A, B, C, or D).
- Provide a clear educational explanation of why the correct option is right based on the notes.
- Do not repeat questions.
- Output MUST be strictly valid JSON without any markdown code fences or conversational text.

JSON FORMAT:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "answer": "A",
      "explanation": "Clear explanation grounded in the text."
    }}
  ]
}}

STUDY MATERIAL:
{context}

TOPIC FOCUS:
{search_query}
"""
                        raw = ask_gemini(prompt)

                        # Clean markdown wrappers if any
                        raw_cleaned = raw.strip()
                        if raw_cleaned.startswith("```"):
                            parts = raw_cleaned.splitlines(True)
                            raw_cleaned = "".join(parts[1:]) if len(parts) > 1 else raw_cleaned
                            if raw_cleaned.endswith("```"):
                                raw_cleaned = raw_cleaned[:-3].strip()
                            if raw_cleaned.lower().startswith("json"):
                                raw_cleaned = raw_cleaned[4:].strip()

                        json_start = raw_cleaned.find("{")
                        json_end = raw_cleaned.rfind("}")
                        if json_start != -1 and json_end != -1:
                            raw_cleaned = raw_cleaned[json_start : json_end + 1]

                        quiz_data = json.loads(raw_cleaned)
                        raw_questions = quiz_data.get("questions", [])

                        normalized_questions = []
                        for item in raw_questions:
                            q_text = item.get("question", "").strip()
                            opts = item.get("options", {})
                            ans = str(item.get("answer", "")).strip().upper()
                            expl = item.get("explanation", "").strip()

                            if isinstance(opts, list):
                                labels = ["A", "B", "C", "D"]
                                opts = {labels[idx]: str(val).strip() for idx, val in enumerate(opts[:4])}
                            elif isinstance(opts, dict):
                                opts = {k.strip().upper(): str(v).strip() for k, v in opts.items()}

                            match = re.match(r"^([A-D])", ans)
                            ans_letter = match.group(1) if match else ans[:1]

                            if q_text and len(opts) >= 2 and ans_letter in opts:
                                normalized_questions.append({
                                    "question": q_text,
                                    "options": opts,
                                    "answer": ans_letter,
                                    "explanation": expl,
                                })

                        if not normalized_questions:
                            st.warning("Could not generate questions. Please try again.")
                        else:
                            st.session_state.quiz_data = normalized_questions
                            st.session_state.quiz_current_index = 0
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_user_answers = {}
                            st.session_state.quiz_score = 0
                            st.rerun()

            except json.JSONDecodeError:
                st.error("The quiz format was invalid. Please click **Create Practice Quiz** again.")
            except Exception as error:
                render_student_error(error, "Quiz Generation")

        questions = st.session_state.get("quiz_data", [])

        if questions:
            # SUBMITTED SCORE & REVIEW SCREEN
            if st.session_state.get("quiz_submitted", False):
                user_answers = st.session_state.get("quiz_user_answers", {})
                score = st.session_state.get("quiz_score", 0)
                total = len(questions)
                percentage = round((score / total) * 100) if total else 0

                encouragement = "🏆 Outstanding performance! Complete mastery demonstrated." if percentage >= 80 else ("👍 Solid effort! Review the explanations below to strengthen your understanding." if percentage >= 50 else "📚 Keep reviewing your notes and try again!")

                st.markdown(
                    f"""
                    <div class="quiz-score-card">
                        <div style="font-size:0.85rem; font-weight:700; color:#4338ca; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Quiz Results</div>
                        <div style="font-family:'Space Grotesk',sans-serif; font-size:3.2rem; font-weight:700; color:#1e1b4b;">{score} / {total}</div>
                        <div style="font-size:1.15rem; font-weight:600; color:#312e81; margin-top:2px;">{percentage}% Score</div>
                        <div style="color:#475569; font-size:0.92rem; margin-top:8px;">{encouragement}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Question Breakdown & Explanations")
                for i, item in enumerate(questions):
                    correct = item.get("answer", "")
                    selected = user_answers.get(i)
                    opts = item.get("options", {})
                    correct_text = opts.get(correct, "")
                    selected_text = opts.get(selected, "") if selected else "Unanswered"
                    is_correct = selected == correct

                    border_color = "#bbf7d0" if is_correct else "#fecaca"
                    bg_color = "#f0fdf4" if is_correct else "#fef2f2"
                    badge_label = "✓ Correct" if is_correct else "✗ Incorrect"
                    badge_color = "#15803d" if is_correct else "#b91c1c"

                    st.markdown(
                        f"""
                        <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:14px; padding:18px 20px; margin-bottom:14px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="font-weight:700; font-size:0.82rem; color:#64748b; text-transform:uppercase;">Question {i + 1} of {total}</span>
                                <span style="font-weight:700; font-size:0.85rem; color:{badge_color};">{badge_label}</span>
                            </div>
                            <div style="font-weight:600; font-size:1.02rem; color:#0f172a; margin-bottom:10px;">{item.get('question')}</div>
                            <div style="font-size:0.9rem; color:#334155; margin-bottom:4px;">
                                <b>Your answer:</b> <span style="color:{'#15803d' if is_correct else '#b91c1c'};">{selected_text} ({selected if selected else 'None'})</span>
                            </div>
                            {f'<div style="font-size:0.9rem; color:#4338ca; margin-bottom:6px;"><b>Correct answer:</b> {correct_text} ({correct})</div>' if not is_correct else ''}
                            <div style="font-size:0.85rem; color:#64748b; margin-top:8px; border-top:1px solid #e2e8f0; padding-top:8px;">
                                💡 <i>{item.get('explanation', '')}</i>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if st.button("🔄 Retake Quiz", use_container_width=True):
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_current_index = 0
                        st.session_state.quiz_user_answers = {}
                        st.session_state.quiz_score = 0
                        st.rerun()
                with col_r2:
                    if st.button("✨ Create New Quiz", type="primary", use_container_width=True):
                        st.session_state.quiz_data = []
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_current_index = 0
                        st.session_state.quiz_user_answers = {}
                        st.session_state.quiz_score = 0
                        st.rerun()

            # ONE QUESTION AT A TIME VIEW
            else:
                idx = st.session_state.get("quiz_current_index", 0)
                total_q = len(questions)
                current_item = questions[idx]

                # Progress Bar
                progress_val = (idx + 1) / total_q
                st.progress(progress_val)

                st.markdown(
                    f"""
                    <div class="quiz-card">
                        <div class="quiz-stepper">
                            <span>Question {idx + 1} of {total_q}</span>
                            <span>{len(st.session_state.quiz_user_answers)} / {total_q} Answered</span>
                        </div>
                        <h3 style="color:#0f172a; font-size:1.15rem; font-weight:600; line-height:1.45; margin-bottom:20px;">
                            {current_item.get('question')}
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                opts = current_item.get("options", {})
                opt_keys = sorted(list(opts.keys()))

                # Current stored choice for this question
                saved_choice = st.session_state.quiz_user_answers.get(idx)
                selected_opt_idx = opt_keys.index(saved_choice) if saved_choice in opt_keys else None

                chosen_answer = st.radio(
                    "Select your answer",
                    opt_keys,
                    format_func=lambda k: f"{k}.  {opts.get(k, '')}",
                    index=selected_opt_idx,
                    key=f"active_quiz_q_{idx}",
                    label_visibility="collapsed",
                )

                if chosen_answer != saved_choice and chosen_answer is not None:
                    st.session_state.quiz_user_answers[idx] = chosen_answer

                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

                col_prev, col_next, col_submit = st.columns([1, 1, 1])

                with col_prev:
                    if idx > 0:
                        if st.button("← Previous", use_container_width=True):
                            st.session_state.quiz_current_index -= 1
                            st.rerun()

                with col_next:
                    if idx < total_q - 1:
                        if st.button("Next Question →", type="primary", use_container_width=True):
                            st.session_state.quiz_current_index += 1
                            st.rerun()

                with col_submit:
                    if st.button("🏁 Finish & Grade Quiz", type="primary" if idx == total_q - 1 else "secondary", use_container_width=True):
                        # Calculate final score
                        final_score = 0
                        for q_i, q_data in enumerate(questions):
                            cor = q_data.get("answer", "").strip().upper()
                            user_ans = st.session_state.quiz_user_answers.get(q_i)
                            if user_ans == cor:
                                final_score += 1

                        st.session_state.quiz_score = final_score
                        st.session_state.quiz_submitted = True
                        st.rerun()
        else:
            if not st.session_state.get("quiz_data"):
                st.markdown(
                    """
                    <div class="ask-start-convo-card">
                        <div class="ask-convo-bubble-icon" style="background:#fff1f2; color:#e11d48;">🎯</div>
                        <div class="ask-convo-head">Ready to start practicing?</div>
                        <p class="ask-convo-desc">
                            Configure your preferred question count and topic to begin.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =========================================================
# 4. SMART SUMMARIES (REVISION GUIDES)
# =========================================================
elif st.session_state.nav_page == "Smart Summaries":
    active_sources = get_indexed_sources(st.session_state.get("vectorstore"))
    doc_count = len(active_sources) if active_sources else 0

    # Top Search Bar
    st.markdown(
        """
        <div class="top-header-row">
            <div class="top-search-container">
                <span class="top-search-icon">🔍</span>
                <input type="text" class="top-search-input" placeholder="Search summaries, key concepts..." />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breadcrumb: Home / Smart Summaries
    st.markdown(
        """
        <div class="ask-breadcrumb">
            <span style="color:#6366f1;">🏠</span>
            <span style="color:#64748b;">Home</span>
            <span style="color:#94a3b8;">/</span>
            <span style="color:#0f172a; font-weight:700;">Smart Summaries</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Header Card: Document Icon, Title, Subtitle & Ready Pill
    doc_pill_label = f"📄 {doc_count} Note{'s' if doc_count != 1 else ''} Attached" if doc_count else "⚡ AI Ready"
    st.markdown(
        f"""
        <div class="ask-header-card">
            <div class="ask-header-left">
                <div class="summary-header-icon-box">📄</div>
                <div>
                    <h1 class="aichat-header-title">Smart Summaries</h1>
                    <p class="aichat-header-subtitle">Turn extensive study materials into clean, well-organized revision notes in a single step.</p>
                </div>
            </div>
            <div>
                <div class="ask-status-pill">
                    <span>{doc_pill_label}</span>
                    <span style="color:#cbd5e1;">•</span>
                    <span style="display:inline-flex; align-items:center; gap:5px; color:#059669; font-weight:700;">
                        <span class="ask-status-dot"></span> Ready
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    if st.session_state.vectorstore is None:
        st.markdown(
            """
            <div class="ask-start-convo-card">
                <div class="ask-convo-bubble-icon" style="background:#eff6ff; color:#2563eb;">📋</div>
                <div class="ask-convo-head">Study notes required for summaries</div>
                <p class="ask-convo-desc">
                    Upload your course PDFs in the left sidebar to generate structured chapter summaries and exam revision guides.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Welcome Banner Card
        st.markdown(
            """
            <div class="summary-welcome-card">
                <div class="aichat-welcome-head">Create high-yield revision summaries in seconds</div>
                <p class="aichat-welcome-desc">
                    Transform long documents into structured study sheets, executive overviews, formula checklists, and core concept breakdowns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Elevated Configuration Card
        with st.expander("⚙️ Summary Settings & Scope", expanded=st.session_state.get("generated_summary") is None):
            col_mode, col_subtopic = st.columns([1, 2])
            with col_mode:
                current_mode_val = st.session_state.get("summary_mode_choice", "Entire study document")
                mode_index = 1 if current_mode_val == "Specific topic" else 0
                summary_mode = st.radio(
                    "Summary Scope",
                    ["Entire study document", "Specific topic"],
                    index=mode_index,
                    key="summary_mode_choice_radio",
                )

            summary_subtopic = ""
            with col_subtopic:
                if summary_mode == "Specific topic":
                    default_sub = st.session_state.get("summary_preset_topic", "")
                    summary_subtopic = st.text_input(
                        "Enter Topic",
                        value=default_sub,
                        placeholder="e.g. ETL vs ELT, Data Warehouse Architectures, Bayes Theorem",
                        key="summary_topic_field",
                    )
                else:
                    st.caption("Consolidates all uploaded material into an organized revision sheet.")

            generate_summary_btn = st.button("📄 Generate Revision Summary", type="primary", use_container_width=True)

        should_gen_summary = generate_summary_btn or st.session_state.get("trigger_summary_gen", False)
        if st.session_state.get("trigger_summary_gen"):
            st.session_state.trigger_summary_gen = False

        if should_gen_summary:
            db = st.session_state.vectorstore

            if summary_mode == "Specific topic":
                clean_top = summary_subtopic.strip() if summary_subtopic.strip() else st.session_state.get("summary_preset_topic", "key concepts")
                with st.spinner("Preparing your revision notes…"):
                    try:
                        docs = retrieve(clean_top, k=8)
                        if not docs:
                            st.warning("No relevant material found for this topic.")
                        else:
                            context = format_context(docs)
                            prompt = f"""You are an expert academic tutor.
Create a comprehensive, well-structured revision study guide on the topic: "{clean_top}".

DOCUMENT SECTIONS REQUIRED:
1. Executive Overview
2. Core Definitions & Principles
3. In-Depth Comparative Breakdown
4. Practical Examples / Application
5. Key Takeaways for Finals Revision

GUIDELINES:
- Base the summary strictly on the provided material without inventing facts.
- Use clean Markdown with headers (##, ###), bullet points, and tables where applicable.

STUDY MATERIAL:
{context}
"""
                            final_summary = ask_gemini(prompt)
                            st.session_state.generated_summary = final_summary
                            st.session_state.summary_scope = f"Topic: {clean_top}"

                    except Exception as error:
                        render_student_error(error, "Summary Generation")
            else:
                with st.spinner("Preparing your complete revision guide…"):
                    try:
                        all_chunks = list(db.docstore._dict.values())
                        if not all_chunks:
                            st.warning("No indexed study content found.")
                        else:
                            unique_texts = []
                            seen_snippets = set()
                            total_chars = 0
                            max_chars = 80000

                            for chunk in all_chunks:
                                text = chunk.page_content.strip()
                                key = text[:100]
                                if key not in seen_snippets:
                                    seen_snippets.add(key)
                                    src = chunk.metadata.get("source", "Document")
                                    page = chunk.metadata.get("page", "?")
                                    formatted_part = f"[{src} - Page {page}]: {text}"
                                    if total_chars + len(formatted_part) < max_chars:
                                        unique_texts.append(formatted_part)
                                        total_chars += len(formatted_part)
                                    else:
                                        break

                            combined_context = chr(10).join([t + chr(10) for t in unique_texts])

                            prompt = f"""You are an expert academic tutor.
Create a comprehensive, cohesive revision study guide from the following document material.

DOCUMENT SECTIONS REQUIRED:
1. Executive Overview
2. Key Definitions & Core Concepts
3. In-Depth Detailed Analysis
4. Important Rules, Formulas, or Patterns
5. Essential Key Takeaways for Fast Revision

GUIDELINES:
- Use clean Markdown with headers (##, ###), bullet points, and bold text for key terms.
- Preserve technical definitions, equations, and critical examples faithfully.
- Do NOT repeat overlapping passages and do not fabricate information.

STUDY MATERIAL:
{combined_context}
"""
                            final_summary = ask_gemini(prompt)
                            st.session_state.generated_summary = final_summary
                            st.session_state.summary_scope = "Entire Study Collection"

                    except Exception as error:
                        render_student_error(error, "Summary Generation")

        # Display Document Summary
        saved_summary = st.session_state.get("generated_summary")
        if saved_summary:
            scope_label = st.session_state.get("summary_scope", "Study Notes")
            word_count = len(saved_summary.split())
            reading_time = max(1, round(word_count / 200))

            st.markdown(
                f"""
                <div class="document-view">
                    <div class="document-header">
                        <div>
                            <span style="font-size:0.75rem; font-weight:700; color:#4f46e5; text-transform:uppercase; letter-spacing:0.08em;">REVISION GUIDE</span>
                            <h3 style="margin:2px 0 0 0; color:#0f172a; font-size:1.35rem;">{scope_label}</h3>
                        </div>
                        <div style="display:flex; gap:10px; font-size:0.8rem; color:#64748b;">
                            <span style="background:#f1f5f9; border:1px solid #e2e8f0; padding:4px 10px; border-radius:6px; color:#475569;">📖 ~{reading_time} min read</span>
                            <span style="background:#f1f5f9; border:1px solid #e2e8f0; padding:4px 10px; border-radius:6px; color:#475569;">📝 {word_count} words</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(saved_summary)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                st.download_button(
                    "📥 Download Revision Guide (.md)",
                    data=saved_summary,
                    file_name="StudyMate_Revision_Guide.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                if st.button("✨ Create New Summary", use_container_width=True):
                    st.session_state.generated_summary = None
                    st.rerun()
        else:
            st.markdown(
                """
                <div class="ask-start-convo-card">
                    <div class="ask-convo-bubble-icon" style="background:#eff6ff; color:#2563eb;">📄</div>
                    <div class="ask-convo-head">No summary generated yet</div>
                    <p class="ask-convo-desc">
                        Configure your summary scope above to generate structured notes.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

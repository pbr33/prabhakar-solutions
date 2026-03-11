"""
Generate a new project estimation Excel in the exact Lenox format.
Sheets: Summary | AI Agent Dev- Backend | DevOps | Architecture | Infra Cost | Version History
All Excel formulas are live (no hard-coded values where formulas existed in the original).
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import date

OUTPUT_FILE = "ECI_AI_Platform_Estimation_v1.0.xlsx"

# ── Palette (exact match to Lenox file) ──────────────────────────────
NAVY        = "FF1F3864"   # dark navy  – section headers
BLUE        = "FF2E75B6"   # medium blue – column headers
LIGHT_BLUE  = "FFDEEAF1"   # light blue – alternate rows
RED         = "FFFF0000"   # red text   – "No"
GREEN       = "FF00B050"   # green text – "Yes"
WHITE       = "FFFFFFFF"
NO_FILL     = "00000000"

def _fill(argb):
    return PatternFill(start_color=argb, end_color=argb, fill_type="solid")

def _font(bold=False, size=11, color="FF000000", name="Calibri", italic=False):
    return Font(name=name, bold=bold, size=size, color=color, italic=italic)

def _thin_border():
    s = Side(style="thin", color="FFAAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def _align(h="left", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

def _hdr_cell(ws, row, col, value=None, fill_argb=NAVY, bold=True,
              font_color="FFFFFFFF", size=11, halign="left", wrap=False):
    c = ws.cell(row=row, column=col)
    if value is not None:
        c.value = value
    c.font = _font(bold=bold, size=size, color=font_color)
    c.fill = _fill(fill_argb)
    c.alignment = _align(h=halign, wrap=wrap)
    return c

def _data_cell(ws, row, col, value=None, bold=False, fill_argb=NO_FILL,
               font_color="FF000000", size=11, halign="left", wrap=False,
               border=True):
    c = ws.cell(row=row, column=col)
    if value is not None:
        c.value = value
    c.font = _font(bold=bold, size=size, color=font_color)
    c.fill = _fill(fill_argb)
    c.alignment = _align(h=halign, wrap=wrap)
    if border:
        c.border = _thin_border()
    return c

def merge_hdr(ws, row, sc, ec, value, fill_argb=NAVY, font_color="FFFFFFFF",
              size=11, bold=True, halign="left"):
    ws.merge_cells(start_row=row, start_column=sc, end_row=row, end_column=ec)
    c = ws.cell(row=row, column=sc)
    c.value = value
    c.font = _font(bold=bold, size=size, color=font_color)
    c.fill = _fill(fill_argb)
    c.alignment = _align(h=halign, wrap=True)
    return c


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: AI Agent Dev- Backend
# ═══════════════════════════════════════════════════════════════════════
def build_backend(wb):
    ws = wb.create_sheet("AI Agent Dev- Backend")
    ws.sheet_view.showGridLines = False

    # --- column widths ---
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 60
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 14
    ws.column_dimensions["F"].width = 14
    ws.column_dimensions["G"].width = 45

    # --- Row 1: Title ---
    merge_hdr(ws, 1, 1, 7,
              "ECI AI PLATFORM BUILD – AI AGENT DEVELOPMENT & INTEGRATION",
              fill_argb=NAVY, size=13, halign="center")
    ws.row_dimensions[1].height = 28

    # --- Row 3: Column headers ---
    for col, val in [(1,"#"),(2,"Task / Phase"),(3,"Sub-task"),
                     (4,"Dev Low (hrs)"),(5,"Dev High (hrs)"),(6,"Avg (hrs)"),(7,"Comments")]:
        _hdr_cell(ws, 3, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[3].height = 18

    # ── Tasks data ──────────────────────────────────────────────────────
    phases = [
        {
            "name": "Phase 1: Discovery & Design",
            "row_range": (5, 13),
            "tasks": [
                (1,  "Project kickoff, scope alignment & stakeholder mapping",                   10, 14, "Workshops with business & tech leads"),
                (2,  "Existing system & data source review (APIs, DBs, file stores)",            8,  12, "Audit current integrations & data quality"),
                (3,  "50–100 document sample analysis (PDFs, Word, Excel)",                      8,  12, "Validate content types and metadata structure"),
                (4,  "Solution architecture finalization & sign-off",                            6,  8,  "Private endpoint, cloud landing zone design"),
                (5,  "Technical design documentation (HLD + LLD)",                              12, 16, "Architecture & data flow diagrams"),
                (6,  "Prompt engineering design for all use cases",                              8,  12, "Use-case specific prompt templates"),
                (7,  "KPI metric & threshold definition",                                        6,  10, "Accuracy, latency, relevance benchmarks"),
                (8,  "Data governance & compliance review",                                      4,  6,  "GDPR / data residency checks"),
                (9,  "Risk assessment & mitigation plan documentation",                          4,  6,  "Risk register creation"),
            ],
        },
        {
            "name": "Phase 2: Data Ingestion & Processing Pipeline",
            "row_range": (16, 24),
            "tasks": [
                (10, "Connector setup for source systems (REST API, SharePoint, SQL)",           8,  12, "One-time ingestion pipeline per source"),
                (11, "PDF/Word/Excel parsing, extraction & metadata tagging",                   10, 14, "Multi-format document processing"),
                (12, "Intelligent chunking strategy & overlap configuration",                    6,  10, "Optimised for RAG retrieval accuracy"),
                (13, "Azure OpenAI embedding integration (2 embedding models)",                 14, 18, "text-embedding-3-large + ada-002"),
                (14, "One-time embedding generation for all source documents",                  12, 16, "Full corpus vectorisation"),
                (15, "Azure AI Search schema design + hybrid search config",                    16, 22, "Vector + BM25 keyword hybrid index"),
                (16, "Azure SQL metadata store design & population",                             6,  10, "Structured lookup tables"),
                (17, "Data validation & ingestion QA",                                          4,  6,  "Spot-check retrieval quality"),
                (18, "Automated re-ingestion trigger (delta updates)",                           6,  10, "Scheduled incremental sync"),
            ],
        },
        {
            "name": "Phase 3: AI Agent Development",
            "row_range": (27, 38),
            "tasks": [
                (19, "Use Case 1: Document Q&A – prompt engineering & RAG pipeline",            10, 14, "Natural language Q&A over enterprise docs"),
                (20, "Use Case 1: Contextual answer generation with citation",                  10, 14, "Source attribution for every response"),
                (21, "Use Case 1: Adaptive Card trigger & response formatting (Teams)",          6,  8,  "Trigger via Teams; output as rich card"),
                (22, "Use Case 2: Intelligent Report Summarisation – RAG pipeline",             10, 14, "Auto-summarise lengthy reports"),
                (23, "Use Case 2: Multi-document synthesis & comparison logic",                  8,  12, "Cross-document insights"),
                (24, "Use Case 2: Structured summary output & export (Word/PDF)",                6,  8,  "Formatted downloadable summaries"),
                (25, "Use Case 3: Anomaly & Insight Detection from structured data",             8,  12, "Flag outliers, trends & exceptions"),
                (26, "Multi-model evaluation – GPT-4o vs GPT-4o-mini + 2 embedding models",    6,  8,  "Compare quality vs cost"),
                (27, "Prompt logging (Azure Monitor / Log Analytics)",                           4,  6,  "Full audit trail"),
                (28, "Security controls – Key Vault, Entra ID managed identity",                4,  8,  "Least-privilege per-user controls"),
                (29, "Rate limiting & user authorisation (10 POC users)",                        4,  6,  "Azure AD group-based access"),
                (30, "Conversation memory & session management",                                 4,  6,  "Short-term context retention"),
            ],
        },
        {
            "name": "Phase 4: Teams Bot Development (Copilot Studio)",
            "row_range": (41, 47),
            "tasks": [
                (31, "Backend REST API (FastAPI) – bot orchestration layer",                    10, 14, "Query routing to AI agent"),
                (32, "Azure AD / Entra SSO integration",                                        4,  6,  "Teams identity & service principal"),
                (33, "Copilot Studio bot config & Teams channel publishing",                     6,  8,  "Teams bot channel deployment"),
                (34, "Adaptive cards / rich response formatting (all 3 use cases)",              6,  8,  "Structured output per use case"),
                (35, "Feedback collection & thumbs-up/down logging",                             4,  6,  "User satisfaction tracking"),
                (36, "End-to-end conversation flow testing",                                     4,  8,  "Validate all use case flows in Teams"),
                (37, "Multi-language support configuration (English + 1 additional)",            4,  6,  "Locale-aware prompts"),
            ],
        },
        {
            "name": "Phase 5: Testing & UAT",
            "row_range": (50, 58),
            "tasks": [
                (38, "Unit testing – all agent components",                                     6,  10, "Function-level tests for all use cases"),
                (39, "Integration testing – end-to-end workflow (all 3 use cases)",             8,  12, "Source → Agent → Teams validation"),
                (40, "Prompt engineering tuning (all use cases)",                               8,  12, "Optimise against KPI thresholds"),
                (41, "Performance testing – response time target <8s",                          4,  6,  "10 concurrent users simulation"),
                (42, "Security testing – auth, authorisation, data isolation",                  4,  6,  "Pen-test checklist"),
                (43, "UAT coordination with 10 users (50–100 queries each)",                    4,  6,  "Real queries across all use cases"),
                (44, "Accessibility & UX review of Teams adaptive cards",                       2,  4,  "WCAG 2.1 AA compliance check"),
                (45, "Feedback incorporation & bug fixes",                                       6,  8,  "Address UAT findings"),
            ],
        },
        {
            "name": "Phase 6: Production Deployment & Handover",
            "row_range": (61, 65),
            "tasks": [
                (46, "Architecture documentation with diagrams (HLD + LLD)",                    4,  6,  "Technical design docs delivered"),
                (47, "User guide creation (with screenshots & video walkthroughs)",             4,  6,  "End-user how-to guide for all use cases"),
                (48, "End-user training sessions (2 × 2 hrs, recorded)",                        4,  4,  "How to use Teams bot for all use cases"),
                (49, "Runbook for ops team (monitoring, alerts, re-ingestion)",                 2,  4,  "Operational handover guide"),
                (50, "Production cutover, smoke testing & go-live sign-off",                    2,  4,  "Final deployment validation"),
            ],
        },
    ]

    # Map phase name → first task row (for formula references later)
    phase_phase_rows = {}   # phase_name → (header_row, sum_row)

    current_row = 4
    for phase in phases:
        ph_name = phase["name"]
        tasks   = phase["tasks"]

        # Phase header row – only merge A:C so D/E/F stay writable for formulas
        ph_header_row = current_row
        task_start = ph_header_row + 1
        task_end   = ph_header_row + len(tasks)
        # Merge only A-C for the phase name label
        ws.merge_cells(start_row=ph_header_row, start_column=1,
                       end_row=ph_header_row, end_column=3)
        c_lbl = ws.cell(ph_header_row, 1)
        c_lbl.value = ph_name
        c_lbl.font  = _font(bold=True, size=10, color="FFFFFFFF")
        c_lbl.fill  = _fill(NAVY)
        c_lbl.alignment = _align(h="left")
        # Formula cells D, E, F
        for col_idx, formula in [
            (4, f"=SUM(D{task_start}:D{task_end})"),
            (5, f"=SUM(E{task_start}:E{task_end})"),
            (6, f"=(D{ph_header_row}+E{ph_header_row})/2"),
        ]:
            c = ws.cell(ph_header_row, col_idx)
            c.value     = formula
            c.font      = _font(bold=True, size=10, color="FFFFFFFF")
            c.fill      = _fill(NAVY)
            c.alignment = _align(h="center")
            c.border    = _thin_border()
        # Style remaining cols
        for col_idx in [7]:
            c = ws.cell(ph_header_row, col_idx)
            c.fill   = _fill(NAVY)
            c.border = _thin_border()
        ws.row_dimensions[ph_header_row].height = 18
        phase_phase_rows[ph_name] = ph_header_row
        current_row += 1

        # Task rows
        for (num, sub, low, high, comment) in tasks:
            _data_cell(ws, current_row, 1, num, halign="center")
            # col B intentionally blank (matches Lenox where phase name only in A)
            _data_cell(ws, current_row, 3, sub, wrap=True)
            _data_cell(ws, current_row, 4, low,  halign="center")
            _data_cell(ws, current_row, 5, high, halign="center")
            ws.cell(current_row, 6).value     = f"=(D{current_row}+E{current_row})/2"
            ws.cell(current_row, 6).font      = _font(size=10)
            ws.cell(current_row, 6).alignment = _align(h="center")
            ws.cell(current_row, 6).border    = _thin_border()
            _data_cell(ws, current_row, 7, comment, wrap=True)
            ws.row_dimensions[current_row].height = 15
            current_row += 1

        current_row += 1  # blank spacer row

    # Project Management row
    pm_row = current_row
    ph_refs_d = "+".join([f"D{r}" for r in phase_phase_rows.values()])
    ph_refs_e = "+".join([f"E{r}" for r in phase_phase_rows.values()])
    merge_hdr(ws, pm_row, 1, 3, "Project Management (10% of dev hours)", fill_argb=BLUE, size=10)
    ws.cell(pm_row, 4).value     = f"=({ph_refs_d})*0.1"
    ws.cell(pm_row, 4).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(pm_row, 4).fill      = _fill(BLUE)
    ws.cell(pm_row, 4).alignment = _align(h="center")
    ws.cell(pm_row, 5).value     = f"=({ph_refs_e})*0.1"
    ws.cell(pm_row, 5).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(pm_row, 5).fill      = _fill(BLUE)
    ws.cell(pm_row, 5).alignment = _align(h="center")
    ws.cell(pm_row, 6).value     = f"=(D{pm_row}+E{pm_row})/2"
    ws.cell(pm_row, 6).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(pm_row, 6).fill      = _fill(BLUE)
    ws.cell(pm_row, 6).alignment = _align(h="center")
    ws.cell(pm_row, 7).value     = "Throughout project"
    ws.cell(pm_row, 7).font      = _font(size=10, color="FFFFFFFF")
    ws.cell(pm_row, 7).fill      = _fill(BLUE)
    ws.row_dimensions[pm_row].height = 18

    # TOTAL row
    total_row = pm_row + 1
    ph_refs_all_d = "+".join([f"D{r}" for r in phase_phase_rows.values()])
    ph_refs_all_e = "+".join([f"E{r}" for r in phase_phase_rows.values()])
    merge_hdr(ws, total_row, 1, 3, "TOTAL PROJECT HOURS", fill_argb=NAVY, size=11)
    ws.cell(total_row, 4).value     = f"={ph_refs_all_d}+D{pm_row}"
    ws.cell(total_row, 4).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 4).fill      = _fill(NAVY)
    ws.cell(total_row, 4).alignment = _align(h="center")
    ws.cell(total_row, 5).value     = f"={ph_refs_all_e}+E{pm_row}"
    ws.cell(total_row, 5).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 5).fill      = _fill(NAVY)
    ws.cell(total_row, 5).alignment = _align(h="center")
    ws.cell(total_row, 6).value     = f"=(D{total_row}+E{total_row})/2"
    ws.cell(total_row, 6).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 6).fill      = _fill(NAVY)
    ws.cell(total_row, 6).alignment = _align(h="center")
    ws.cell(total_row, 7).value     = "Average hours noted for reference"
    ws.cell(total_row, 7).font      = _font(bold=False, size=10, color="FFFFFFFF")
    ws.cell(total_row, 7).fill      = _fill(NAVY)
    ws.row_dimensions[total_row].height = 20

    return ws, total_row  # return so Summary can reference F{total_row}


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: DevOps
# ═══════════════════════════════════════════════════════════════════════
def build_devops(wb):
    ws = wb.create_sheet("DevOps")
    ws.sheet_view.showGridLines = False

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 14
    ws.column_dimensions["F"].width = 14
    ws.column_dimensions["G"].width = 45

    merge_hdr(ws, 1, 1, 7,
              "ECI AI PLATFORM BUILD – DEVOPS & INFRASTRUCTURE PROVISIONING",
              fill_argb=NAVY, size=13, halign="center")
    ws.row_dimensions[1].height = 28

    for col, val in [(1,"#"),(2,"Task / Phase"),(3,"Sub-task"),
                     (4,"Dev Low (hrs)"),(5,"Dev High (hrs)"),(6,"Avg (hrs)"),(7,"Comments")]:
        _hdr_cell(ws, 3, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[3].height = 18

    sections = [
        {
            "name": "Azure Environment Setup",
            "tasks": [
                (1,  "Azure Resource Group, VNet & Subnets provisioning",               2,  3,  "Isolated network foundation"),
                (2,  "Private endpoints & Private Link config (all services)",           8,  12, "All services off public internet"),
                (3,  "Azure AI Foundry workspace setup & model deployments",             4,  6,  "GPT-4o + GPT-4o-mini + embedding models"),
                (4,  "Azure AI Search instance (Standard S1) provisioning",             2,  4,  "Vector + keyword hybrid search"),
                (5,  "Azure App Service (P2V3) for bot backend",                        2,  3,  "REST API hosting for FastAPI backend"),
                (6,  "Azure Key Vault & secrets configuration",                         2,  4,  "API keys, credentials, managed identity"),
                (7,  "Azure Monitor + Log Analytics (90-day retention)",                2,  3,  "Prompt logging & observability"),
                (8,  "Azure SQL Database (Standard S3) provisioning",                   3,  4,  "Structured metadata & audit log storage"),
                (9,  "Azure Blob Storage (LRS) for document staging",                   2,  3,  "Temp staging & processed doc archive"),
                (10, "SharePoint & REST API service principal configuration",            3,  4,  "Auth for all data source connectors"),
                (11, "Entra ID / Azure AD app registrations",                           2,  3,  "Auth for APIs & Teams bot"),
                (12, "Azure Front Door / API Management (optional WAF layer)",           3,  5,  "Rate limiting, WAF rules"),
            ],
        },
        {
            "name": "Copilot Studio & Teams Integration",
            "tasks": [
                (13, "Copilot Studio environment provisioning & bot channel setup",     3,  5,  "Teams bot channel configuration"),
                (14, "Teams admin consent & bot deployment to tenant",                  3,  5,  "Admin approval required"),
            ],
        },
        {
            "name": "Security Baseline & Testing",
            "tasks": [
                (15, "NSG rules, Azure AD conditional access, Key Vault policies",      4,  6,  "Security hardening per scope"),
                (16, "UAT staging environment deployment",                               2,  3,  "Mirror of production"),
                (17, "RBAC configuration – least-privilege roles per team member",       2,  3,  "PIM-based access review"),
            ],
        },
    ]

    section_header_rows = []
    current_row = 4

    for section in sections:
        sec_header_row = current_row
        task_start = current_row + 1
        task_end   = current_row + len(section["tasks"])
        section_header_rows.append(sec_header_row)

        # Merge only A-C for label, leave D/E/F for formulas
        ws.merge_cells(start_row=sec_header_row, start_column=1,
                       end_row=sec_header_row, end_column=3)
        c_lbl = ws.cell(sec_header_row, 1)
        c_lbl.value = section["name"]
        c_lbl.font  = _font(bold=True, size=10, color="FFFFFFFF")
        c_lbl.fill  = _fill(NAVY)
        c_lbl.alignment = _align(h="left")
        for col_idx, formula in [
            (4, f"=SUM(D{task_start}:D{task_end})"),
            (5, f"=SUM(E{task_start}:E{task_end})"),
            (6, f"=(D{sec_header_row}+E{sec_header_row})/2"),
        ]:
            c = ws.cell(sec_header_row, col_idx)
            c.value     = formula
            c.font      = _font(bold=True, size=10, color="FFFFFFFF")
            c.fill      = _fill(NAVY)
            c.alignment = _align(h="center")
            c.border    = _thin_border()
        ws.cell(sec_header_row, 7).fill   = _fill(NAVY)
        ws.cell(sec_header_row, 7).border = _thin_border()
        ws.row_dimensions[sec_header_row].height = 18
        current_row += 1

        for (num, sub, low, high, comment) in section["tasks"]:
            _data_cell(ws, current_row, 1, num, halign="center")
            _data_cell(ws, current_row, 3, sub, wrap=True)
            _data_cell(ws, current_row, 4, low,  halign="center")
            _data_cell(ws, current_row, 5, high, halign="center")
            ws.cell(current_row, 6).value     = f"=(D{current_row}+E{current_row})/2"
            ws.cell(current_row, 6).font      = _font(size=10)
            ws.cell(current_row, 6).alignment = _align(h="center")
            ws.cell(current_row, 6).border    = _thin_border()
            _data_cell(ws, current_row, 7, comment, wrap=True)
            ws.row_dimensions[current_row].height = 15
            current_row += 1

        current_row += 1  # spacer

    # Meetings & Sync (10%)
    mtg_row = current_row
    sec_d = "+".join([f"D{r}" for r in section_header_rows])
    sec_e = "+".join([f"E{r}" for r in section_header_rows])
    merge_hdr(ws, mtg_row, 1, 3, "Meetings & Sync (10% of dev hours)", fill_argb=BLUE, size=10)
    ws.cell(mtg_row, 4).value     = f"=({sec_d})*0.1"
    ws.cell(mtg_row, 4).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(mtg_row, 4).fill      = _fill(BLUE)
    ws.cell(mtg_row, 4).alignment = _align(h="center")
    ws.cell(mtg_row, 5).value     = f"=({sec_e})*0.1"
    ws.cell(mtg_row, 5).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(mtg_row, 5).fill      = _fill(BLUE)
    ws.cell(mtg_row, 5).alignment = _align(h="center")
    ws.cell(mtg_row, 6).value     = f"=(D{mtg_row}+E{mtg_row})/2"
    ws.cell(mtg_row, 6).font      = _font(bold=True, size=10, color="FFFFFFFF")
    ws.cell(mtg_row, 6).fill      = _fill(BLUE)
    ws.cell(mtg_row, 6).alignment = _align(h="center")
    ws.cell(mtg_row, 7).value     = "Standups, client syncs"
    ws.cell(mtg_row, 7).font      = _font(size=10, color="FFFFFFFF")
    ws.cell(mtg_row, 7).fill      = _fill(BLUE)
    ws.row_dimensions[mtg_row].height = 18

    # TOTAL row
    total_row = mtg_row + 1
    merge_hdr(ws, total_row, 1, 3, "TOTAL DEVOPS HOURS", fill_argb=NAVY, size=11)
    ws.cell(total_row, 4).value     = f"={sec_d}+D{mtg_row}"
    ws.cell(total_row, 4).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 4).fill      = _fill(NAVY)
    ws.cell(total_row, 4).alignment = _align(h="center")
    ws.cell(total_row, 5).value     = f"={sec_e}+E{mtg_row}"
    ws.cell(total_row, 5).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 5).fill      = _fill(NAVY)
    ws.cell(total_row, 5).alignment = _align(h="center")
    ws.cell(total_row, 6).value     = f"=(D{total_row}+E{total_row})/2"
    ws.cell(total_row, 6).font      = _font(bold=True, size=11, color="FFFFFFFF")
    ws.cell(total_row, 6).fill      = _fill(NAVY)
    ws.cell(total_row, 6).alignment = _align(h="center")
    ws.cell(total_row, 7).value     = "Average for budgeting"
    ws.cell(total_row, 7).font      = _font(size=10, color="FFFFFFFF")
    ws.cell(total_row, 7).fill      = _fill(NAVY)
    ws.row_dimensions[total_row].height = 20

    return ws, total_row


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: Architecture
# ═══════════════════════════════════════════════════════════════════════
def build_architecture(wb):
    ws = wb.create_sheet("Architecture")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["D"].width = 90

    merge_hdr(ws, 1, 1, 7, "ECI AI PLATFORM BUILD – SOLUTION ARCHITECTURE",
              fill_argb=NAVY, size=13, halign="center")
    ws.row_dimensions[1].height = 28

    # Use case descriptions (matching Lenox layout – D25:D26 style)
    ws.cell(25, 4).value = (
        "Use Case 1 – Document Q&A · Natural language question answering over enterprise documents "
        "(PDFs, Word, Excel) via Azure AI Search + GPT-4o RAG pipeline · Teams Adaptive Card trigger → Text + citation output"
    )
    ws.cell(25, 4).alignment = _align(wrap=True)

    ws.cell(26, 4).value = (
        "Use Case 2 – Intelligent Report Summarisation · Automated multi-document synthesis & comparison "
        "with structured downloadable summary (Word/PDF) · Teams agent search → Formatted summary output"
    )
    ws.cell(26, 4).alignment = _align(wrap=True)

    ws.cell(27, 4).value = (
        "Use Case 3 – Anomaly & Insight Detection · Structured data analysis flagging outliers, trends "
        "and exceptions across connected data sources · Teams notification → Dashboard-ready insight cards"
    )
    ws.cell(27, 4).alignment = _align(wrap=True)

    ws.row_dimensions[25].height = 48
    ws.row_dimensions[26].height = 48
    ws.row_dimensions[27].height = 48

    # Architecture notes
    notes = [
        (3, "Platform",     "Microsoft Azure (existing subscription leveraged)"),
        (4, "AI Engine",    "Azure OpenAI – GPT-4o (primary) + GPT-4o-mini (cost optimisation)"),
        (5, "Vector Store", "Azure AI Search Standard S1 – hybrid (vector + BM25)"),
        (6, "Embeddings",   "text-embedding-3-large + text-embedding-ada-002"),
        (7, "Bot Channel",  "Copilot Studio → Microsoft Teams"),
        (8, "Backend API",  "FastAPI on Azure App Service P2V3"),
        (9, "Auth",         "Azure AD / Entra ID – managed identity, SSO"),
        (10,"Data Sources", "SharePoint Online, REST APIs, Azure SQL, Azure Blob Storage"),
        (11,"Monitoring",   "Azure Monitor + Log Analytics (90-day retention)"),
        (12,"Security",     "Azure Key Vault, Private Endpoints, NSG, Conditional Access"),
    ]
    for row, label, value in notes:
        ws.cell(row, 1).value = label
        ws.cell(row, 1).font  = _font(bold=True, size=10)
        ws.cell(row, 2).value = value
        ws.cell(row, 2).font  = _font(size=10)
        ws.row_dimensions[row].height = 16

    return ws


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: Infra Cost
# ═══════════════════════════════════════════════════════════════════════
def build_infra_cost(wb):
    ws = wb.create_sheet("Infra Cost")
    ws.sheet_view.showGridLines = False

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 48
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 22
    ws.column_dimensions["E"].width = 65

    merge_hdr(ws, 1, 1, 5,
              "ECI AI PLATFORM BUILD – INFRASTRUCTURE COST ESTIMATE",
              fill_argb=NAVY, size=13, halign="center")
    ws.row_dimensions[1].height = 28

    for col, val in [(1,"#"),(2,"Service / Resource"),(3,"$/mo"),(4,"Pricing Model"),(5,"Notes / Assumptions")]:
        _hdr_cell(ws, 2, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[2].height = 18

    # Cost line items
    sections = [
        {
            "heading": "  AI Services — Azure AI Foundry / OpenAI",
            "items": [
                (1,  "GPT-4o (Primary LLM — 85% of queries)",          35,  "Pay-per-token",      "~750 queries/mo avg 1,200 input + 600 output tokens. GPT-4o: $2.50/$10 per 1M tokens. ~85% of queries."),
                (2,  "GPT-4o-mini (Cost-optimised — 15% of queries)",   5,   "Pay-per-token",      "~135 queries/mo comparison testing. GPT-4o-mini: $0.15/$0.60 per 1M tokens."),
            ],
            "subtotal_label": "Subtotal — AI Services",
        },
        {
            "heading": "  Vector Storage — Azure AI Search",
            "items": [
                (3,  "Azure AI Search — Standard S1 (1 replica, 1 partition)",  245, "Fixed monthly",  "S1 required for semantic ranker + hybrid search. ~$245/mo base. Supports all 3 use case indexes."),
            ],
            "subtotal_label": "Subtotal — Vector Storage",
        },
        {
            "heading": "  Compute — App Service & Functions",
            "items": [
                (4,  "Azure App Service — P2V3 Linux (FastAPI backend / Teams bot)",  80,  "Fixed monthly",      "P2V3: 2 vCPU, 8 GB RAM. ~$80/mo. Hosts REST API for Copilot Studio bot."),
                (5,  "Azure Functions — Consumption Plan (ingestion helper)",           5,  "Pay-per-execution",  "Used during SharePoint ingestion & delta sync. Near-zero ongoing cost."),
                (6,  "Azure Blob Storage — LRS (document staging & archive)",          10,  "Pay-per-GB",         "~20 GB document corpus + staging. $0.018/GB/mo."),
            ],
            "subtotal_label": "Subtotal — Compute",
        },
        {
            "heading": "  Database — Azure SQL",
            "items": [
                (7,  "Azure SQL Database — Standard S3 (100 DTUs)",  150,  "Fixed monthly",  "S3 = 100 DTUs. Metadata queries, audit logs and cap rate lookup tables. Private endpoint included."),
            ],
            "subtotal_label": "Subtotal — Database / Azure SQL",
        },
        {
            "heading": "  Security & Identity",
            "items": [
                (8,  "Azure Key Vault (secrets, API keys, managed identity)",          10,   "Operations-based",     "Standard tier. ~15K operations/mo. First 10K ops free, then $0.03/10K."),
                (9,  "Entra ID / Azure AD (service principals, app registrations)",    0,    "Included in M365 E3",  "No additional cost. 10 POC users on existing E3 licences."),
                (10, "Azure Front Door / API Management (WAF + rate limiting)",        50,   "Fixed monthly",        "Developer tier API Management. WAF policy + IP filtering."),
            ],
            "subtotal_label": "Subtotal — Security & Identity",
        },
        {
            "heading": "  Monitoring & Observability",
            "items": [
                (11, "Azure Monitor — Log Analytics Workspace",   35,  "Per GB ingested",  "Prompt logs for all interactions. ~3-5 GB/mo. $2.30/GB after 5 GB free tier."),
                (12, "Azure Monitor — Alerts & Dashboards",        5,  "Per alert rule",   "10 alert rules for latency, error rate, ingestion status."),
            ],
            "subtotal_label": "Subtotal — Monitoring & Observability",
        },
        {
            "heading": "  Integration — Copilot Studio & Teams",
            "items": [
                (13, "Copilot Studio (Teams bot, Adaptive Cards – all 3 use cases)",  200,  "Pay per message / tenant",  "~$200/tenant/mo standalone OR depends on M365 licence tier. Confirm with client."),
            ],
            "subtotal_label": "Subtotal — Integration",
        },
    ]

    current_row = 3
    subtotal_rows = []

    for section in sections:
        # Section heading
        merge_hdr(ws, current_row, 1, 5, section["heading"], fill_argb=NAVY, size=10)
        ws.row_dimensions[current_row].height = 16
        item_start = current_row + 1
        current_row += 1

        for (num, service, cost, model, notes) in section["items"]:
            _data_cell(ws, current_row, 1, num, halign="center")
            _data_cell(ws, current_row, 2, service)
            c = ws.cell(current_row, 3)
            c.value = cost
            c.font  = _font(size=10)
            c.alignment = _align(h="center")
            c.border = _thin_border()
            _data_cell(ws, current_row, 4, model)
            _data_cell(ws, current_row, 5, notes, wrap=True)
            ws.row_dimensions[current_row].height = 30
            current_row += 1

        item_end = current_row - 1

        # Subtotal row
        sub_row = current_row
        subtotal_rows.append(sub_row)
        merge_hdr(ws, sub_row, 1, 2, section["subtotal_label"], fill_argb=LIGHT_BLUE,
                  font_color="FF000000", size=10, bold=True)
        ws.cell(sub_row, 3).value     = f"=SUM(C{item_start}:C{item_end})"
        ws.cell(sub_row, 3).font      = _font(bold=True, size=10)
        ws.cell(sub_row, 3).fill      = _fill(LIGHT_BLUE)
        ws.cell(sub_row, 3).alignment = _align(h="center")
        ws.cell(sub_row, 3).border    = _thin_border()
        for col in [4, 5]:
            ws.cell(sub_row, col).fill   = _fill(LIGHT_BLUE)
            ws.cell(sub_row, col).border = _thin_border()
        ws.row_dimensions[sub_row].height = 16
        current_row += 2  # spacer after each section

    # GRAND TOTAL row
    total_row = current_row
    total_formula = "+".join([f"C{r}" for r in subtotal_rows])
    merge_hdr(ws, total_row, 1, 2, "TOTAL ESTIMATED MONTHLY INFRASTRUCTURE COST",
              fill_argb=NAVY, size=11, halign="left")
    ws.cell(total_row, 3).value     = f"={total_formula}"
    ws.cell(total_row, 3).font      = _font(bold=True, size=12, color="FFFFFFFF")
    ws.cell(total_row, 3).fill      = _fill(NAVY)
    ws.cell(total_row, 3).alignment = _align(h="center")
    ws.cell(total_row, 3).border    = _thin_border()
    ws.cell(total_row, 4).fill      = _fill(NAVY)
    note_cell = ws.cell(total_row, 5)
    note_cell.value     = "Estimated monthly cost for POC environment. Excludes one-time setup costs and Copilot Studio (confirm licence with client)."
    note_cell.font      = _font(size=9, color="FFFFFFFF", italic=True)
    note_cell.fill      = _fill(NAVY)
    note_cell.alignment = _align(wrap=True)
    ws.row_dimensions[total_row].height = 28

    return ws, total_row, subtotal_rows


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: Summary
# ═══════════════════════════════════════════════════════════════════════
def build_summary(wb, backend_total_row, devops_total_row, infra_total_row):
    ws = wb.active
    ws.title = "Summary"
    ws.sheet_view.showGridLines = False

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 5
    ws.column_dimensions["F"].width = 6
    ws.column_dimensions["G"].width = 22
    ws.column_dimensions["H"].width = 14
    ws.column_dimensions["I"].width = 12
    ws.column_dimensions["J"].width = 5
    ws.column_dimensions["K"].width = 5
    ws.column_dimensions["L"].width = 6
    ws.column_dimensions["M"].width = 75

    # ── Row 1: Title ──────────────────────────────────────────────────
    merge_hdr(ws, 1, 1, 13,
              "ECI AI PLATFORM BUILD – ESTIMATION SUMMARY",
              fill_argb=NAVY, size=14, halign="center")
    ws.row_dimensions[1].height = 30

    # ── Row 3: Section headers ────────────────────────────────────────
    _hdr_cell(ws, 3, 1, "Variables",            fill_argb=NAVY, size=10)
    ws.merge_cells("A3:E3")
    _hdr_cell(ws, 3, 6, "Team Allocation",       fill_argb=NAVY, size=10)
    ws.merge_cells("F3:J3")
    _hdr_cell(ws, 3, 12, "S.N.  |  Pre-Requisites (Client Responsibility)", fill_argb=NAVY, size=10)
    ws.merge_cells("L3:M3")
    ws.row_dimensions[3].height = 18

    # ── Row 4: Column headers ─────────────────────────────────────────
    for col, val in [(1,"SNO"),(2,"Variables"),(3,"#"),(4,""),(5,"")]:
        _hdr_cell(ws, 4, col, val, fill_argb=BLUE, size=10, halign="center")
    for col, val in [(6,"SNO"),(7,"Role"),(8,"Allocation %"),(9,"Est. Days"),(10,"")]:
        _hdr_cell(ws, 4, col, val, fill_argb=BLUE, size=10, halign="center")
    for col, val in [(12,""),(13,"")]:
        _hdr_cell(ws, 4, col, val, fill_argb=LIGHT_BLUE, size=10)
    ws.row_dimensions[4].height = 18

    # ── Variables ─────────────────────────────────────────────────────
    variables = [
        (1, "FrontEnd", "No",  RED),
        (2, "Backend",  "Yes", GREEN),
        (3, "DevOps",   "Yes", GREEN),
        (4, "QA",       "Yes", GREEN),
        (5, "Mobile",   "No",  RED),
    ]
    for i, (sno, name, yn, yn_color) in enumerate(variables):
        row = 5 + i
        _data_cell(ws, row, 1, sno,  halign="center")
        _data_cell(ws, row, 2, name)
        c = ws.cell(row, 3)
        c.value = yn
        c.font  = _font(bold=True, size=10, color="FFFFFFFF" if yn_color == GREEN else "FFFFFFFF")
        c.font  = _font(bold=True, size=10, color="FFFFFFFF")
        c.fill  = _fill(yn_color)
        c.alignment = _align(h="center")
        c.border = _thin_border()
        ws.row_dimensions[row].height = 15

    # ── Activities label ──────────────────────────────────────────────
    ws.cell(10, 1).value = "Activities"
    ws.cell(10, 1).font  = _font(bold=True, size=10)
    ws.row_dimensions[10].height = 16

    # Activities columns
    for col, val in [(1,"SNO"),(2,"Activities"),(4,"DevTime (hrs)")]:
        _hdr_cell(ws, 11, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[11].height = 16

    activities = [
        (1, "FrontEnd",   "0"),
        (2, "Backend",    f"='AI Agent Dev- Backend'!F{backend_total_row}"),
        (3, "DevOps",     f"=DevOps!F{devops_total_row}"),
    ]
    for i, (sno, name, formula) in enumerate(activities):
        row = 12 + i
        _data_cell(ws, row, 1, sno, halign="center")
        _data_cell(ws, row, 2, name)
        c = ws.cell(row, 4)
        c.value     = formula
        c.font      = _font(size=10)
        c.alignment = _align(h="center")
        c.border    = _thin_border()
        ws.row_dimensions[row].height = 15

    # Monthly infra cost ref
    ws.cell(14, 6).value     = "Monthly Infra Cost (Est.)"
    ws.cell(14, 6).font      = _font(bold=True, size=10)
    ws.merge_cells("F14:G14")
    ws.cell(14, 8).value     = f"='Infra Cost'!C{infra_total_row}"
    ws.cell(14, 8).font      = _font(bold=True, size=10)
    ws.cell(14, 8).alignment = _align(h="center")
    ws.cell(14, 8).border    = _thin_border()
    ws.row_dimensions[14].height = 16

    # ── Component summary table ────────────────────────────────────────
    for col, val in [(2,"Component"),(3,"Days"),(4,"Weeks")]:
        _hdr_cell(ws, 16, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[16].height = 16

    comp_rows = [
        (17, "FrontEnd", "0",
              "0"),
        (18, "Backend",  f"='AI Agent Dev- Backend'!F{backend_total_row}/7",
              f"='AI Agent Dev- Backend'!F{backend_total_row}/35"),
        (19, "DevOps",   f"=DevOps!F{devops_total_row}/7",
              f"=DevOps!F{devops_total_row}/35"),
    ]
    for (row, comp, days_f, weeks_f) in comp_rows:
        _data_cell(ws, row, 2, comp)
        c_d = ws.cell(row, 3)
        c_d.value     = days_f
        c_d.font      = _font(size=10)
        c_d.alignment = _align(h="center")
        c_d.border    = _thin_border()
        c_w = ws.cell(row, 4)
        c_w.value     = weeks_f
        c_w.font      = _font(size=10)
        c_w.alignment = _align(h="center")
        c_w.border    = _thin_border()
        ws.row_dimensions[row].height = 15

    # Project Duration
    ws.cell(20, 1).value = "Project Duration"
    ws.cell(20, 1).font  = _font(bold=True, size=10)
    ws.merge_cells("A20:C20")
    c_dur = ws.cell(20, 4)
    c_dur.value     = f"='AI Agent Dev- Backend'!F{backend_total_row}/35"
    c_dur.font      = _font(bold=True, size=10)
    c_dur.alignment = _align(h="center")
    c_dur.border    = _thin_border()
    ws.row_dimensions[20].height = 16

    # Duration with leaves
    ws.cell(22, 1).value = "Project Duration including Leaves & Holiday"
    ws.cell(22, 1).font  = _font(bold=True, size=10)
    ws.merge_cells("A22:C22")
    ws.row_dimensions[22].height = 16

    _hdr_cell(ws, 23, 2, "Leaves & Holiday", fill_argb=BLUE, size=10)
    ws.cell(23, 3).value = 1  # 1 week buffer
    ws.cell(23, 3).font  = _font(size=10)
    ws.cell(23, 3).alignment = _align(h="center")
    ws.cell(23, 3).border = _thin_border()
    _data_cell(ws, 23, 4, "Weeks")

    c_tdur = ws.cell(24, 3)
    c_tdur.value     = f"='AI Agent Dev- Backend'!F{backend_total_row}/35+C23"
    c_tdur.font      = _font(bold=True, size=10)
    c_tdur.alignment = _align(h="center")
    c_tdur.border    = _thin_border()
    _data_cell(ws, 24, 4, "Weeks")
    ws.row_dimensions[24].height = 15

    ws.cell(25, 1).value = "Note: With 2 AI Engineers, project can be completed within 8-10 weeks."
    ws.cell(25, 1).font  = _font(italic=True, size=9, color="FF555555")
    ws.merge_cells("A25:J25")
    ws.row_dimensions[25].height = 16

    # ── Team Allocation ───────────────────────────────────────────────
    for col, val in [(6,"SNO"),(7,"Role"),(8,"Allocation %"),(9,"Est. Days")]:
        _hdr_cell(ws, 5, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[5].height = 16

    team = [
        (1, "Project Manager",        "25%", 0.25),
        (2, "AI Architect / Lead",     "50%", 0.50),
        (3, "AI Engineer",             "100%",1.00),
        (4, "QA Engineer",             "50%", 0.50),
        (5, "DevOps Engineer",         "25%", 0.25),
        (6, "Data Engineer",           "50%", 0.50),
    ]
    base_f = f"('AI Agent Dev- Backend'!F{backend_total_row}+DevOps!F{devops_total_row})/7"
    for i, (sno, role, alloc_str, alloc) in enumerate(team):
        row = 6 + i
        _data_cell(ws, row, 6, sno, halign="center")
        _data_cell(ws, row, 7, role)
        _data_cell(ws, row, 8, alloc_str, halign="center")
        c = ws.cell(row, 9)
        c.value     = f"=({base_f})*{alloc}"
        c.font      = _font(size=10)
        c.alignment = _align(h="center")
        c.border    = _thin_border()
        ws.row_dimensions[row].height = 15

    # ── Pre-Requisites ────────────────────────────────────────────────
    prereqs = [
        "Active Azure subscription with Contributor or Owner role at resource group level",
        "Microsoft 365 E3 (or higher) licenses for all service and user accounts",
        "Copilot Studio license (standalone ~$200/tenant/mo or via M365 tier — confirm with client)",
        "Read/write SharePoint credentials and Graph API permissions for all source data",
        "Microsoft Teams admin consent for custom Copilot Studio bot deployment",
        "Azure AD user accounts provisioned for 10 POC users",
        "Identification of key stakeholders for workshops and UAT sign-off",
        "Access to all REST APIs and data sources listed in scope",
        "Sample document corpus (50–100 files per use case) provided within 1 week of kickoff",
        "Agreed document template formats locked before ingestion begins",
    ]
    # Row 3 headers for L:M already set above (merged L3:M3)
    for i, pr in enumerate(prereqs):
        row = 4 + i
        _data_cell(ws, row, 12, i+1, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 13, pr, wrap=True, fill_argb=LIGHT_BLUE)
        ws.row_dimensions[row].height = 20

    # ── Out of Scope ──────────────────────────────────────────────────
    oos_start = 4 + len(prereqs) + 2
    merge_hdr(ws, oos_start - 1, 12, 13, "S.N.  |  Out of Scope",
              fill_argb=NAVY, size=10)
    ws.row_dimensions[oos_start - 1].height = 18

    out_of_scope = [
        "Data ingestion from sources outside the agreed list (non-approved data)",
        "Automated real-time ingestion — only one-time + scheduled delta ingestion in scope",
        "Custom UI development outside Teams (no web portal or mobile app)",
        "Building full CI/CD pipelines or advanced release automation",
        "PowerPoint (.pptx) and image files as primary data source types",
        "Long-term AI operations support beyond the 30-day stabilisation period",
        "Unstructured formats like audio and video not supported as data sources",
        "Multi-language support beyond English + 1 additional language",
        "Real-time external data feeds (market data, live APIs beyond what is listed)",
        "Any new features or requests not mentioned in scope (change request process applies)",
    ]
    for i, oos in enumerate(out_of_scope):
        row = oos_start + i
        _data_cell(ws, row, 12, i+1, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 13, oos, wrap=True, fill_argb=LIGHT_BLUE)
        ws.row_dimensions[row].height = 20

    # ── Assumptions ───────────────────────────────────────────────────
    assump_start = oos_start + len(out_of_scope) + 2
    merge_hdr(ws, assump_start - 1, 12, 13, "S.N.  |  Assumptions",
              fill_argb=NAVY, size=10)
    ws.row_dimensions[assump_start - 1].height = 18

    assumptions = [
        "Source files are PDF, Word, and Excel only — all other formats excluded unless specifically agreed",
        "50–100 historical files per use case will be provided within POC scope",
        "AI response accuracy is estimated at 80–95% for structured queries. 100% accuracy is not guaranteed.",
        "AI responses are advisory in nature — all outputs must be reviewed and validated by client before decisions.",
        "Model accuracy is dependent on quality, consistency, and completeness of source files provided.",
        "A single agreed-upon document template per use case will be confirmed and locked during Discovery & Design.",
        "One-time ingestion plus scheduled delta sync; no real-time streaming ingestion.",
        "10 users will participate in UAT, each testing 50–100 queries across all 3 use cases.",
        "Users must have existing SharePoint and Teams access to use the agent.",
        "Prompt logs stored in Azure Monitor / Log Analytics with 90-day default retention.",
        "Existing Azure landing zone and subscription will be leveraged — no new subscription required.",
        "Client is responsible for all Azure/cloud licensing, infrastructure, and data usage costs.",
        "Client will assign UAT resource(s) and respond to ECI queries within 24–48 hours.",
        "The UAT phase will follow a set timeline; no new enhancements during UAT.",
        "Data classification and labeling not required for the POC phase.",
    ]
    for i, assump in enumerate(assumptions):
        row = assump_start + i
        _data_cell(ws, row, 12, i+1, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 13, assump, wrap=True, fill_argb=LIGHT_BLUE)
        ws.row_dimensions[row].height = 20

    # ── Risks ─────────────────────────────────────────────────────────
    risk_start = assump_start + len(assumptions) + 2
    merge_hdr(ws, risk_start - 1, 12, 13, "S.N.  |  Risks",
              fill_argb=NAVY, size=10)
    ws.row_dimensions[risk_start - 1].height = 18

    risks = [
        "API restrictions: Permission or access issues may add significant effort — validate in Discovery.",
        "LLM hallucination risk: Without human review, incorrect AI outputs could influence real decisions.",
        "File volume unvalidated: 50–100 files/use case is an estimate; insufficient data may impact KPI validation.",
        "Document inconsistency: Varying formats across analysts will degrade AI accuracy.",
        "Scope creep: Use cases may expand to additional reports or data types during UAT — change request process applies.",
        "Model performance variability: GPT-4o vs GPT-4o-mini quality gap may require additional prompt tuning.",
        "Delta sync complexity: Incremental ingestion logic may require additional effort if source structure changes.",
        "Copilot Studio licensing: Cost depends on client M365 tier — standalone ~$200/tenant/mo; confirm before start.",
        "Prompt log retention: Retention beyond 90 days requires additional configuration and may impact cost.",
        "Teams deployment approval: Bot deployment to tenant requires admin consent — delays possible.",
    ]
    for i, risk in enumerate(risks):
        row = risk_start + i
        _data_cell(ws, row, 12, i+1, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 13, risk, wrap=True, fill_argb=LIGHT_BLUE)
        ws.row_dimensions[row].height = 20

    # ── Definition of Done ────────────────────────────────────────────
    dod_start = risk_start + len(risks) + 2
    merge_hdr(ws, dod_start - 1, 12, 13, "S.N.  |  Definition of Done – Project Closure Criteria",
              fill_argb=NAVY, size=10)
    ws.row_dimensions[dod_start - 1].height = 18

    dod_sections = {
        "  ► DISCOVERY & DESIGN": [
            "Technical architecture document reviewed and signed off by client stakeholders",
            "KPI metrics and acceptance thresholds defined, documented, and agreed upon by both parties",
            "All data source access confirmed and tested (SharePoint, REST APIs, SQL)",
            "Prompt templates provided and approved for all 3 use cases",
        ],
        "  ► ENVIRONMENT SETUP": [
            "All Azure resources provisioned: Resource Group, VNet, AI Foundry, AI Search, App Service, Key Vault, Azure Monitor",
            "Entra ID / Azure AD app registrations and service principals created with least-privilege access",
            "Copilot Studio bot deployed to client's Teams tenant with admin consent granted",
        ],
        "  ► DATA INGESTION": [
            "One-time ingestion completed for all 3 use cases — files validated and indexed in Azure AI Search",
            "Two embedding models tested; embeddings stored and validated",
            "Hybrid search (vector + BM25) confirmed functional via representative test queries",
            "Azure SQL metadata store populated and validated",
            "Delta sync trigger configured and tested",
        ],
        "  ► AI AGENT": [
            "All 3 use case pipelines returning relevant responses meeting KPI thresholds",
            "Multi-model comparison completed: GPT-4o vs GPT-4o-mini evaluated across defined KPIs",
            "Prompt logging active — all interactions captured in Azure Monitor (90-day retention confirmed)",
            "Adaptive Card trigger and Teams bot responses functional for all 3 use cases",
        ],
        "  ► TESTING & UAT": [
            "All unit and integration tests passing with minimum critical defects",
            "10 POC users completed UAT (50–100 queries each) across all 3 use cases",
            "Security testing passed: authentication, authorization, data isolation verified",
            "Feedback incorporated and sign-off obtained from UAT participants",
        ],
        "  ► PRODUCTION DEPLOYMENT & HANDOVER": [
            "Production environment deployed, smoke tested, and monitoring alerts configured",
            "Architecture documentation and user guides finalized and delivered to client",
            "Production access credentials transferred with least-privilege RBAC",
            "Runbook delivered to operations team",
        ],
    }

    row = dod_start
    for section_title, items in dod_sections.items():
        ws.cell(row, 12).value = section_title
        ws.cell(row, 12).font  = _font(bold=True, size=10, italic=True)
        ws.cell(row, 12).fill  = _fill(LIGHT_BLUE)
        ws.cell(row, 13).fill  = _fill(LIGHT_BLUE)
        ws.merge_cells(f"L{row}:M{row}")
        ws.row_dimensions[row].height = 16
        row += 1
        for i, item in enumerate(items, 1):
            _data_cell(ws, row, 12, i, halign="center", fill_argb=LIGHT_BLUE)
            _data_cell(ws, row, 13, item, wrap=True, fill_argb=LIGHT_BLUE)
            ws.row_dimensions[row].height = 20
            row += 1

    return ws


# ═══════════════════════════════════════════════════════════════════════
#  SHEET: Version History
# ═══════════════════════════════════════════════════════════════════════
def build_version_history(wb):
    ws = wb.create_sheet("Version History")
    ws.sheet_view.showGridLines = False

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 15
    ws.column_dimensions["D"].width = 20

    merge_hdr(ws, 1, 1, 4, "VERSION HISTORY", fill_argb=NAVY, size=13, halign="center")
    ws.row_dimensions[1].height = 25

    for col, val in [(1,"Version"),(2,"Changes"),(3,"Date"),(4,"Author")]:
        _hdr_cell(ws, 3, col, val, fill_argb=BLUE, size=10, halign="center")
    ws.row_dimensions[3].height = 18

    versions = [
        ("v1.0", "Initial estimate based on ECI AI Platform scope document",
         str(date.today()), "Prabhakar"),
    ]
    for i, (ver, changes, dt, author) in enumerate(versions):
        row = 4 + i
        _data_cell(ws, row, 1, ver, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 2, changes, wrap=True, fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 3, dt, halign="center", fill_argb=LIGHT_BLUE)
        _data_cell(ws, row, 4, author, fill_argb=LIGHT_BLUE)
        ws.row_dimensions[row].height = 18

    return ws


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    wb = openpyxl.Workbook()

    # Build detail sheets first so we know their total-row numbers
    ws_backend, backend_total_row = build_backend(wb)
    ws_devops,  devops_total_row  = build_devops(wb)
    ws_arch                       = build_architecture(wb)
    ws_infra,   infra_total_row, _= build_infra_cost(wb)
    ws_ver                        = build_version_history(wb)

    # Build Summary last — it references the other sheets
    build_summary(wb, backend_total_row, devops_total_row, infra_total_row)

    # Reorder sheets so Summary is first (it was created as wb.active = first)
    # openpyxl keeps insertion order, Summary is already ws[0]
    # Move it to position 0 just in case
    wb.move_sheet("Summary", offset=-(wb.sheetnames.index("Summary")))

    wb.save(OUTPUT_FILE)
    print(f"✅  Saved: {OUTPUT_FILE}")
    print(f"   Backend total row : {backend_total_row}")
    print(f"   DevOps total row  : {devops_total_row}")
    print(f"   Infra total row   : {infra_total_row}")
    print(f"   Sheets            : {wb.sheetnames}")


if __name__ == "__main__":
    main()

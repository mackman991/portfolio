# NHS Capacity & Resource Utilisation

Diagnostic analysis of NHS appointment data spanning 42 Integrated Care Boards
and 106 sub-locations from 2020 to 2022, covering the full COVID-19 disruption
period and recovery.

LSE Data Analytics Career Accelerator — Course 2 assignment.

## Key findings

- The NHS's 1.2m daily appointment guideline was breached on **175 of 334 days** analysed
- Identified clear seasonal peaks (winter pressure), regional capacity disparities across ICBs,
  and the rapid pivot to telephone consultations during the first lockdown
- Telephone/online appointments rose sharply in Q1 2020 and remained structurally elevated
- Built scenario analysis modelling the impact of **3% vs 6% missed-appointment (DNA) rates**
  on daily capacity — demonstrated that reducing DNAs would push average volumes toward or
  above operational thresholds, requiring parallel workforce planning
- Delivered recommendations on data-quality standards, regional resource allocation and
  capacity planning

## Files

| File | Description |
|---|---|
| `Mackin_Peter_LSE_DA201_Assignment_Notebook.ipynb` | Full analysis notebook (Python) |
| `Mackin_Peter_DA201_Assignment_Report.pdf` | Written report submitted to LSE |
| `Mackin_Peter_DA201_Assignment_slides.pdf` | Presentation slides |

## Tools & methods

- **Python** — pandas, matplotlib, seaborn
- **Tableau** — regional and time-series dashboards
- Diagnostic / exploratory analysis
- Scenario modelling
- Time-series decomposition (seasonal patterns, COVID disruption)

## Setup

```bash
pip install pandas matplotlib seaborn jupyter
jupyter notebook Mackin_Peter_LSE_DA201_Assignment_Notebook.ipynb
```

Data sourced from NHS England open datasets (appointment activity by ICB).

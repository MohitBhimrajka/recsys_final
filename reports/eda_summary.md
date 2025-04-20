# Exploratory Data Analysis (EDA) Summary - OULAD Dataset

Date: 2025-04-20

## 1. Overview

This document summarizes the initial findings from exploring the Open University Learning Analytics Dataset (OULAD). The analysis covers the structure, distributions, missing values, and basic relationships within the key data tables: `studentInfo`, `courses`, `studentRegistration`, `studentVle`, `vle`, `assessments`, and `studentAssessment`. All raw data files were successfully located and loaded.

## 2. Data Files Summary

- **assessments.csv:** (206 rows, 6 columns). Defines assessments, types (TMA, CMA, Exam), dates (11 missing), weights. Assessment types are TMA (106), CMA (76), Exam (24). Weights range from 0 to 100, with many assessments having 0 weight.
- **courses.csv:** (22 rows, 3 columns). Defines 7 unique modules and their 22 presentations. Presentation lengths range from 234 to 269 days, typically around 260 days.
- **studentAssessment.csv:** (173,912 rows, 5 columns). Records student submissions for assessments. Contains `score` (173 missing values), `date_submitted`, and `is_banked` (mostly 0). Scores range from 0 to 100, mean ~75.8, std ~18.8.
- **studentInfo.csv:** (32,593 rows, 12 columns). Demographic and registration summary data per student per presentation. Contains `final_result`. `imd_band` has 1111 missing values.
- **studentRegistration.csv:** (32,593 rows, 5 columns). Logs student registration/unregistration dates. `date_registration` has 45 missing values. `date_unregistration` has 22,521 missing values (approx. 69% missing), indicating most students did not officially unregister. No duplicate registrations found for the same student in the same presentation.
- **studentVle.csv:** (10,655,280 rows, 6 columns). Detailed logs of student interactions (26,074 unique students, 6,268 unique VLE items). **Largest table.** No missing values reported in core columns.
- **vle.csv:** (6,364 rows, 6 columns). Metadata about VLE materials (6,364 unique `id_site`). `week_from` and `week_to` have 5,243 missing values each (approx. 82.4% missing). Defines 20 unique `activity_type`s.

## 3. Key Findings & Observations

### 3.1. Student Demographics (`studentInfo`)

- **Gender:** Skewed towards Male (M: ~17.9k, F: ~14.7k). *(Correction: The provided output shows M=17875, F=14718, so slightly more males, but reasonably balanced).*
- **Region:** Scotland and East Anglian Region are most common. Wide distribution across 13 regions.
- **Highest Education:** 'A Level or Equivalent' (14,045) is the most common, followed by 'Lower Than A Level' (11,163) and 'HE Qualification' (6,263). 'No Formal quals' and 'Post Graduate Qualification' are less common.
- **IMD Band:** Missing for 1111 records (~3.4%). Otherwise spread across bands, with '20-30%' being the most frequent non-missing band. -> *Action: Imputation (e.g., median/mode) or treat as a separate category.*
- **Age Band:** Dominated by '0-35' (22,944), followed by '35-55' (9,433). '55<=' is rare (216).
- **Disability:** Majority 'N' (No - 29,429). 'Y' (Yes - 3,164).
- **Previous Attempts:** Heavily skewed towards 0 (~84%). Max attempts is 6.
- **Studied Credits:** Most students take 60 or 120 credits. Some take modules with >200 credits. Distribution shows peaks around common credit loads (30, 60, 90, 120).
- **Final Result:** 'Pass' (12,361) is most common, followed by 'Withdrawn' (10,156), 'Fail' (7,052), and 'Distinction' (3,024). Significant number of 'Withdrawn' students.

### 3.2. Courses & Presentations (`courses`)

- **Modules vs Presentations:** 7 unique modules (AAA-GGG), 22 unique presentations (e.g., 2013J, 2014J). Modules BBB, DDD, FFF have 4 presentations each; AAA, CCC have 2.
- **Presentation Length:** Most presentations are long (~260-270 days), with a few shorter ones (~240 days).

### 3.3. Registrations (`studentRegistration`)

- **Registration Timing:** Distribution is roughly normal, centered around day -69 (mean). Students register significantly before the official start (day 0), ranging from ~1 year early (-322) to ~half a year late (167).
- **Unregistration:** Occurs for ~30.9% of registrations. Mean unregistration date is ~day 50. Distribution peaks early after the start date. Many missing values indicate students simply stop participating rather than formally unregistering. -> *Unregistration date is a strong signal, but missingness needs careful handling.*

### 3.4. VLE Interactions (`studentVle`, `vle`)

- **Interaction Volume:** Massive dataset (10.6M records). 26,074 unique students logged. Average student has ~409 interaction records (median 270), but highly variable (std 430).
- **Clicks (`sum_click`):** Average clicks *per record* is low (~3.7), median is 2. Highly skewed (max 6977). Suggests most interaction log entries represent few clicks. Total clicks *per student* average ~1519 (median 824), also highly variable (std 1935). -> *Log transformation needed for visualization/modeling. Defining meaningful interaction level is key (e.g., total clicks > threshold, distinct days interacted).*
- **Interaction Timing:** Activity peaks around the start of the course and then gradually decreases, with smaller peaks likely corresponding to assessment periods.
- **VLE Item Types (`activity_type`):** Dominated by 'resource' (2660), 'subpage' (1063), 'oucontent' (991), 'url' (845), 'forumng' (527). Many other types are less frequent. -> *Good source for content features.*
- **Interaction Sparsity:** While the total volume is high, the distribution plots (log scale) show many students with relatively fewer interactions/clicks compared to the highly active ones.

### 3.5. Assessments & Scores (`assessments`, `studentAssessment`)

- **Assessment Types:** TMAs (Tutor Marked Assignments) are most common (106), followed by CMAs (Computer Marked - 76) and Exams (24).
- **Weights:** Many assessments have 0 weight, potentially formative. Weights range up to 100 (likely Exams). Distribution shows peaks at 0 and around common percentages.
- **Scores:** Mean score is ~75.8 (median 80), indicating generally high performance among those submitting. 173 missing scores need handling. Distribution is left-skewed (tail towards lower scores).
- **Pass Rate:** Very high pass rate (95.5%) based on a simple score >= 40 threshold on submitted assessments.
- **Submission Timing (`days_early`):** Mean submission is ~16.7 days early (median 1 day early). Distribution is peaked around 0, with a long tail for early submissions and a shorter tail for late submissions (up to -372, possibly data errors or very early deadlines?).

### 3.6. Initial Relationships

- **Demographics vs. Score:** Higher education levels correlate with higher average assessment scores (Post Grad > HE Qual > A Level > Lower Than A Level > No Formal Quals).
- **Interaction vs. Outcome:** Boxplots (log scale, outliers hidden) show a clear positive correlation between both `total_clicks` and `total_interactions` per student/presentation and their `final_result`. Students achieving 'Distinction' or 'Pass' have significantly more interactions/clicks than those who 'Fail' or are 'Withdrawn'. -> *Strongly supports using interaction data (clicks, frequency) as implicit feedback for recommendations.*

## 4. Data Quality Issues & Next Steps

- **Missing Values:**
    - `studentInfo.imd_band` (~3.4%): Impute (mode?) or use a separate category.
    - `studentRegistration.date_unregistration` (~69%): High missingness is informative (student didn't formally withdraw). Create a 'withdrawn_flag' feature or handle carefully.
    - `studentAssessment.score` (173 records): Relatively few. Could drop these records or impute (e.g., mean/median score for that assessment, though potentially biased).
    - `assessments.date` (11 records): Affects deadline calculation. Need to decide how to handle assessments without deadlines or impute if possible.
    - `vle.week_from`/`week_to` (~82%): Too much missing data to rely heavily on weekly VLE structure for most items. Can still use for items where available.
    - `studentRegistration.date_registration` (45 records): Drop these few records or impute.
- **Outliers:** High `sum_click` values observed. Log transform or clipping might be needed for some modeling approaches. `days_early` has extreme negative values (-372) - investigate or cap.
- **ID Merging:** Need consistent creation and use of `presentation_id` (`code_module` + `code_presentation`) across tables.
- **Implicit Feedback:** Define strategy. Total `sum_click` per student per presentation seems promising, potentially log-transformed or binned. Alternatively, count of distinct interaction days or distinct VLE items accessed.
- **Filtering:** Need to set thresholds based on EDA (e.g., min interaction records/clicks per user like 5 or 10, min registered/interacting users per presentation like 5 or 10) to handle sparsity. Distributions (e.g., interactions per student) suggest many low-activity users.
- **Time Split:** Interaction dates span a wide range. Need to choose a cutoff date (e.g., `2014-09-01` as initially proposed in `config.py` seems reasonable, need to verify against presentation dates) that provides sufficient data for training and a meaningful test set.

## 5. Potential Features for Models

- **User Features:** One-hot encoded demographics (gender, region, highest_education, age_band, disability, possibly imd_band), `num_of_prev_attempts`, `studied_credits`, maybe aggregated performance features (avg score, pass rate).
- **Item Features (Presentation):** `module_presentation_length`, one-hot encoded `code_module`, counts/proportions of different `activity_type`s in the VLE, counts/types of assessments (TMA, CMA, Exam), average weight of assessments.
- **Interaction Features (Implicit Feedback):** Log-transformed `total_clicks` or `total_interactions` per (student, presentation), count of distinct interaction days, count of distinct VLE items accessed per presentation. Assessment scores could also be incorporated as stronger implicit feedback.
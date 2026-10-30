# Submit revised Weeks 2 to 6

Name: **M Nithish Kumar**

Repository URL to paste in every remaining task:

https://github.com/nithishchowdary29/beauty-review-intelligence

Week 1 already has a recorded evaluation. This package does not require you
to delete it or resubmit it. Its original report remains a historical file.
Weeks 2–6 now contain real public data analysis and supersede their initial
synthetic versions.

## Download

Open the repository, click **Code > Download ZIP**, right-click the downloaded
ZIP in File Explorer, select **Extract All**, then open the extracted folder.
The Word files are in `reports`. The ZIP is a delivery bundle, not the report
file to upload to the internship portal.

## Submit one week at a time

| Portal task | What it asks for | Word file |
|---|---|---|
| Week 2 | Source evaluation, public links, collection plan and quality assessment | `Week_02_Data_Sourcing.docx` |
| Week 3 | Missing values, duplicates, outliers, preprocessing tools and flowchart | `Week_03_Data_Cleaning.docx` |
| Week 4 | EDA methods, charts, features and business interpretation | `Week_04_EDA_Features.docx` |
| Week 5 | At least two models, rationale, validation, metrics and pitfalls | `Week_05_Modeling.docx` |
| Week 6 | Evaluation framework, actual results, visualizations and recommendations | `Week_06_Evaluation_Impact.docx` |

For each week:

1. Read the matching Word report and open its evidence links.
2. Open `Submission_Descriptions.docx` or `.txt` and find that week's section.
3. In the Yuvaintern weekly form, select **Choose File** and upload the
   matching `.docx`. Do not upload the ZIP, Python file or description document.
4. Paste the repository URL shown above into **GitHub project URL**.
5. Copy that week's description body into the description text box. Exclude
   the heading and word-count label. Each body exceeds 200 words.
6. Review the file name, URL and text. Click **Submit** yourself.
7. Confirm a submitted status or confirmation message. Save your own record.

## What you should understand before submitting

The public collection has 371345 reviews. We uniformly sampled 30000 using
seed 42. Metadata matching removes 441 rows, missing text removes 31 and exact
normalized duplicates remove 1919, leaving 27609. Ratings 1–2 are the positive
label. Product IDs and ratings are not predictors. Test products are absent
from training. Logistic text was selected by validation average precision.
The reports discuss the actual mistakes and limitations, not only successes.

Run the commands in README.md if possible. Do not claim that you collected
the original reviews, worked certified hours, deployed a live service or
measured customer savings. The project uses a historical public research
source. Evaluation acceptance and score remain the provider's decision.

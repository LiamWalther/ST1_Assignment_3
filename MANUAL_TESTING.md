# Manual Testing Evidence
Unit: Software Technology 1 (4483/8995) — Assignment 3

## How to run tests
All tests are performed manually by running the console application and
entering the inputs described below. Evidence is provided via screenshots
saved alongside this document.

---

## Test Scenarios

| # | Scenario | Input | Expected Result | Actual Result | Evidence |
|---|----------|-------|-----------------|---------------|----------|
| 1 | Missing dataset folder | Run app with empty `data/raw/` | Friendly error: "Dataset folder not found" | ✅ Passed | Screenshot_01.png |
| 2 | Invalid image path for prediction | Enter `wrong_file.jpg` at predict prompt | Error: "Could not read image" | ✅ Passed | Screenshot_02.png |
| 3 | Predict before training | Choose option 4 before option 3 | Error: "No trained model found. Run option 3 first" | ✅ Passed | Screenshot_03.png |
| 4 | Invalid menu option | Enter `9` or `abc` at menu | "Invalid option. Please enter a number between 1 and 5" | ✅ Passed | Screenshot_04.png |
| 5 | Blank input at predict prompt | Press Enter with no path | "No path entered. Returning to menu." | ✅ Passed | Screenshot_05.png |
| 6 | Dataset summary — valid data | Choose option 1 with dataset loaded | Prints total images, classes, dimensions | ✅ Passed | Screenshot_06.png |
| 7 | EDA generation — valid data | Choose option 2 | Three PNG files saved to `outputs/eda/` | ✅ Passed | Screenshot_07.png |
| 8 | Model training — valid data | Choose option 3 | Model saved, accuracy printed, report saved | ✅ Passed | Screenshot_08.png |
| 9 | Valid prediction | Choose option 4 with real image path | Predicted class and confidence shown | ✅ Passed | Screenshot_09.png |
| 10 | Clean exit | Choose option 5 | "Exiting. Goodbye!" printed, program ends | ✅ Passed | Screenshot_10.png |

---

## Notes
- Screenshots are stored in `outputs/reports/test_screenshots/`
- All tests were run on mac os with Python 3.13
- Dataset used: Kaggle Stream Macroinvertebrates dataset

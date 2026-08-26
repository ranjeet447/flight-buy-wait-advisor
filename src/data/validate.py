"""src/data/validate.py

Automated Data Quality & Schema Validation Suite using Great Expectations.
"""

from pathlib import Path
import sys
import great_expectations as gx
import pandas as pd

RAW_DATA_PATH = Path("data/raw/Clean_Dataset.csv")


def run_data_validation(df: pd.DataFrame) -> bool:
  print("🔍 Running Great Expectations Data Validation Suite...")

  # Wrap pandas DataFrame in a Great Expectations dataset
  context = gx.get_context()
  data_source = context.data_sources.add_pandas("pandas_flight_source")
  data_asset = data_source.add_dataframe_asset(name="flight_raw_asset")

  batch_def = data_asset.add_batch_definition_whole_dataframe("batch_def")
  batch = batch_def.get_batch(batch_parameters={"dataframe": df})

  # Create an Expectation Suite
  suite = context.suites.add(
      gx.ExpectationSuite(name="flight_data_quality_suite")
  )

  # 1. Essential Columns Must Exist
  required_columns = [
      "airline",
      "flight",
      "source_city",
      "departure_time",
      "stops",
      "arrival_time",
      "destination_city",
      "class",
      "duration",
      "days_left",
      "price",
  ]
  suite.add_expectation(
      gx.expectations.ExpectTableColumnsToMatchSet(
          column_set=required_columns, exact_match=False
      )
  )

  # 2. No Null Values in Critical Decision Features
  for col in ["airline", "source_city", "destination_city", "days_left", "price"]:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
    )

  # 3. Valid Value Ranges & Categories
  valid_cities = [
      "Delhi",
      "Mumbai",
      "Bangalore",
      "Kolkata",
      "Hyderabad",
      "Chennai",
  ]
  suite.add_expectation(
      gx.expectations.ExpectColumnValuesToBeInSet(
          column="source_city", value_set=valid_cities
      )
  )
  suite.add_expectation(
      gx.expectations.ExpectColumnValuesToBeInSet(
          column="destination_city", value_set=valid_cities
      )
  )

  # Price must be realistic for Indian domestic travel (₹1,000 to ₹150,000)
  suite.add_expectation(
      gx.expectations.ExpectColumnValuesToBeBetween(
          column="price", min_value=1000, max_value=150000
      )
  )

  # Days left must be between 1 and 60
  suite.add_expectation(
      gx.expectations.ExpectColumnValuesToBeBetween(
          column="days_left", min_value=1, max_value=60
      )
  )

  # Run Validation
  validation_definition = context.validation_definitions.add(
      gx.ValidationDefinition(
          name="flight_validation_def", data=batch_def, suite=suite
      )
  )

  results = validation_definition.run(batch_parameters={"dataframe": df})

  if results.success:
    print("✅ Great Expectations: All data validation assertions PASSED!")
    return True
  else:
    print("❌ Great Expectations: Data validation FAILED!")
    print(results)
    return False


if __name__ == "__main__":
  if not RAW_DATA_PATH.exists():
    print(f"Error: Data file not found at {RAW_DATA_PATH}")
    sys.exit(1)

  raw_df = pd.read_csv(RAW_DATA_PATH)
  success = run_data_validation(raw_df)
  if not success:
    sys.exit(1)
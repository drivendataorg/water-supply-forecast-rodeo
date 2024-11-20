import os
import argparse
from datetime import datetime

def generate_forecast(issue_date, site_id, target_dir):
    # Define the site short names
    site_short = {
        'owyhee_r_bl_owyhee_dam': 'owyhee',
        'pueblo_reservoir_inflow': 'pueblo',
    }

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Create or overwrite the _variables.yml file with the current site and date
    variables_path = os.path.join(target_dir, '_variables.yml')
    with open(variables_path, 'w') as file:
        file.write(f"site: {site_id}\n")
        file.write(f"site_short: {site_short[site_id]}\n")
        file.write(f"issue_date: {issue_date}\n")

    # Create the command to run quarto with the current site and date
    forecast_pdf = f"forecast-{site_short[site_id]}-{issue_date}.pdf"
    command = f"quarto render {target_dir}/forecast.qmd --to pdf -o {forecast_pdf} "
    print(f"Executing command: {command}")

    # Execute the command
    os.system(command)
    os.system(f"mv {forecast_pdf} {target_dir}")
    print(f"Generated forecast PDF: {target_dir}/{forecast_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Generate forecast PDFs")
    parser.add_argument("--issue_date", required=True, help="Issue date in YYYY-MM-DD format")
    parser.add_argument("--site_id", required=True, choices=['owyhee_r_bl_owyhee_dam', 'pueblo_reservoir_inflow'], help="Site ID")
    parser.add_argument("--target_dir", default=".", help="Target directory for output files")

    args = parser.parse_args()

    # Validate the date format
    try:
        datetime.strptime(args.issue_date, '%Y-%m-%d')
    except ValueError:
        print("Error: issue_date must be in YYYY-MM-DD format")
        return

    generate_forecast(args.issue_date, args.site_id, args.target_dir)

if __name__ == "__main__":
    main()

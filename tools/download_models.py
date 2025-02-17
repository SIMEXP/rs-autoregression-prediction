import os

import pandas as pd
from comet_ml import API
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    api = API(os.getenv("COMET_API_TOKEN"))
    experiments = api.get_experiments(
        os.getenv("WORKSPACE"), project_name=os.getenv("PROJECT_NAME")
    )
    for exp in experiments:
        exp_name = exp.get_name()
        metrics = pd.DataFrame(exp.get_metrics())
        metrics.to_csv(
            f"outputs/performance_info/{exp_name}_metrics.tsv",
            sep="\t",
            index=False,
        )
        if len(exp.get_asset_list()) > 2:
            for deet in exp.get_asset_list():
                if "ckpt" in deet["fileName"]:
                    curl_command = (
                        deet["curlDownload"].split(" > ")[0]
                        + " > outputs/model_registery/"
                        + exp_name
                        + ".ckpt"
                    )
                    print(curl_command)
                    # metrics = pd.DataFrame(exp.get_metrics())
                    # metrics.to_csv(f"outputs/performance_info/{exp_name}_metrics.tsv", sep='\t', index=False)

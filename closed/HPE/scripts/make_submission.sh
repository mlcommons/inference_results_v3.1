# This script runs NVIDIA code to export and check the submission
# refer to documentation/submission_guide.md for more details

SUBMITTER="HPE"

export SUBMITTER=$SUBMITTER
cd closed/$SUBMITTER/ #container's home dir (called `/work` inside container)

#Truncate results - Run this outside of container
####wait to truncate until you're ready to submit
#make truncate_results SUBMITTER=$SUBMITTER # << only need to run once
cp -r build/full_results/ ../full_results-backup #only needed if you want another copy

make copy_results_artifacts SUBMITTER=$SUBMITTER
make export_submission SUBMITTER=$SUBMITTER

make check_submission SUBMITTER=$SUBMITTER
make pack_submission SUBMITTER=$SUBMITTER


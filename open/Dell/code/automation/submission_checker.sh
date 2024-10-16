set -x
submission=$1
submitter=$2
new_submission=${submission}_new

if [ -d ${new_submission} ];then 
	rm -rf ${new_submission}
fi

cd <INFERENCE_REPO>
python3 ./tools/submission/truncate_accuracy_log.py --input $submission --submitter $submitter --output ${new_submission} 
#skip compliance for Bfloat16
python3 ./tools/submission/submission_checker.py --input ${new_submission}  --submitter $submitter --version v3.1
set +x

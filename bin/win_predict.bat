echo on
@Rem model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt, multiclass_linear)
set model_name=???

set config_path=config/model/%model_name%.conf

@Rem data file for predicting
set file_name=???

@Rem train/test line python transform switch & script
set transform="false"
set transform_script_path=bin/transform.py

@Rem result save mode: PREDICT_RESULT_ONLY, LABEL_AND_PREDICT, PREDICT_AS_FEATURE
set resultSaveMode=PREDICT_RESULT_ONLY
set resultFileSuffix=_%model_name%_%resultSaveMode%

@Rem max error data format tolerate number
set max_error_tol=100

set eval_metric="auc,mae"

@Rem value or leafid
set predict_type=value

cd ..
@start /b cmd /c java -Xmx1000m -XX:-OmitStackTraceInFastThrow -cp lib/* -Dlog4j.configuration=file:config/log4j_win.properties com.fenbi.ytklearn.predictor.Predicts %config_path% %model_name% %file_name% %transform% %transform_script_path% %resultSaveMode% %resultFileSuffix% %max_error_tol% %eval_metric% %predict_type%

pause
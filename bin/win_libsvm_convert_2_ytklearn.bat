echo on
@Rem binary_classification@label1,label2, multi_classification@label1,label2,..., regression
set mode= ???

set x_delim="###"
set y_delim=","
set features_delim=","
set feature_name_val_delim=":"

set fs_scheme="???"
set libsvm_data_path="???"
set ytklearn_data_path="???"

@start /b cmd /c java -Xmx1000m -XX:-OmitStackTraceInFastThrow -cp lib/* -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.utils.LibsvmConvertTool %mode% %x_delim% %y_delim% %features_delim% %feature_name_val_delim% %fs_scheme% %libsvm_data_path% %ytklearn_data_path%

pause
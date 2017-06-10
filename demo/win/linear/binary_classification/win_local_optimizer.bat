echo on
@Rem thread number
set thread_num=6

@Rem use current machine as master
FOR /F "usebackq" %%i IN (`hostname`) DO SET master_host=%%i
echo %master_host%

@Rem if you run more than one training tasks on the same host at the same time,
@Rem different tasks must have different ports!
set master_port=61235

@Rem model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt, multiclass_linear)
set model_name=linear

@Rem model config
set properties_path=demo/win/linear/binary_classification/%model_name%.conf

@Rem train/test line python transform switch & script
set transform=false
set transform_script_path=bin/transform.py

cd ../../../../
@Rem start ytk-mp4j master, default max memory is 512m
@start /b cmd /c java -Xmx512m -cp lib/* -Dlog4j.configuration=file:config/log4j_master_win.properties com.fenbi.mp4j.comm.CommMaster 1 %master_port%

@Rem start windows train local worker, default max memory is 1000m
@start /b cmd /c java -Xmx1000m -XX:-OmitStackTraceInFastThrow -cp lib/* -Dlog4j.configuration=file:config/log4j_slave_win.properties com.fenbi.ytklearn.worker.LocalTrainWorker  %model_name% %properties_path% %transform_script_path% %transform% user %master_host% %master_port% %thread_num%

pause

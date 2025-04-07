@echo off
setlocal

@REM 定义参数列表
set params="Ukulele_01" "antique_katana_01" "boulder_01" "chinese_chandelier" "dartboard" "Drill_01" "dry_branches_medium_01" "food_kiwi_01" "hand_plane_no4" "power_box_01" "sofa_03" "Television_01" "lambis_shell" "lubricant_spray" "treasure_chest" "metal_trash_can"
set params1="antique_katana_01" "boulder_01" "chinese_chandelier" "dartboard" "Drill_01" "dry_branches_medium_01" "food_kiwi_01" "hand_plane_no4" "power_box_01" "sofa_03" "Television_01" "lambis_shell" "lubricant_spray" "treasure_chest" "metal_trash_can"

set run_mode="2"
set epoch="4000"
set lr="0.01"
set quantizeMode="3"
set optimizeMode="1"
set encode_config_selection_Type="1"
set pretain="0"
set nm_vaild="0"
set objectname="lubricant_spray"
set Fix_DBC_best_epoch="600"
set DBC_best_epoch="1600"
set nm_codec_name="BC7"
set Ns="2"
set Nr="2"
set featuresize="512"
set log="1"

@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "0" %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "2" "2" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "0" "4" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "0" "8" %featuresize% %log%

@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "0" "4" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "4" "0" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "0" %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "0" "2" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "1" "1" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "2" "0" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "2" "2" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "0" "8" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "4" "4" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "8" "0" %featuresize% %log%
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" %pretain% %nm_vaild% %objectname% %Fix_DBC_best_epoch% %DBC_best_epoch% %nm_codec_name% "64" "0" %featuresize% %log%

@REM train
@REM (for %%p in (%params1%) do (
@REM     call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "0" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "512" %log%
@REM     call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "0" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "512" %log%
@REM ))
@REM (for %%p in (%params%) do (
@REM     call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC7" "2" "2" "512" %log%
@REM ))
(for %%p in (%params%) do (
    @REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "0" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "1024" %log%
    @REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "0" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "1024" %log%
    call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC7" "2" "2" "1024" %log%
))
(for %%p in (%params%) do (
    call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "0" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "2048" %log%
    call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "0" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC6" "2" "2" "2048" %log%
    call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %epoch% %lr% %quantizeMode% "1" "1" "1" %nm_vaild% %%p %Fix_DBC_best_epoch% %DBC_best_epoch% "BC7" "2" "2" "2048" %log%
))
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "Ukulele_01" "1000" "900" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "antique_katana_01" "600" "800" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "boulder_01" "700" "850" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "chinese_chandelier" "1250" "900" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "dartboard" "400" "700" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "Drill_01" "1450" "1050" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "dry_branches_medium_01" "950" "900" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "food_kiwi_01" "550" "850" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "hand_plane_no4" "1000" "950" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "power_box_01" "600" "800" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "sofa_03" "350" "850" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "Television_01" "600" "750" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "lambis_shell" "1850" "1400" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "lubricant_spray" "1050" "850" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "treasure_chest" "750" "800" %nm_codec_name% "0"
@REM call CMakeLibTorch_examples_4_NeuralSolver.exe  %run_mode% %refinecount% %epoch% %lr% %leaky% %optimizeMode% %encode_config_selection_Type% "0" "1" "metal_trash_can" "2200" "1000" %nm_codec_name% "0"

endlocal
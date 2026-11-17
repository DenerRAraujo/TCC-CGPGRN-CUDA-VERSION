python3 include/makeFile.py -o 1 -n 500 -e 50000
bash mCAD_exe_1_Cluster0_pt0.sh;
bash mCAD_exe_2_Cluster0_pt0.sh;
bash mCAD_exe_3_Cluster0_pt0.sh;
bash mCAD_exe_4_Cluster0_pt0.sh;
bash mCAD_exe_5_Cluster0_pt0.sh;
bash mCAD_exe_1_Cluster1_pt0.sh;
bash mCAD_exe_2_Cluster1_pt0.sh;
bash mCAD_exe_3_Cluster1_pt0.sh;
bash mCAD_exe_4_Cluster1_pt0.sh;
bash mCAD_exe_5_Cluster1_pt0.sh;
python3 include/makeFile.py -o 1 -n 500 -e 50000
bash mCAD_exe_1_Cluster0_pt1.sh;
bash mCAD_exe_2_Cluster0_pt1.sh;
bash mCAD_exe_3_Cluster0_pt1.sh;
bash mCAD_exe_4_Cluster0_pt1.sh;
bash mCAD_exe_5_Cluster0_pt1.sh;
bash mCAD_exe_1_Cluster1_pt1.sh;
bash mCAD_exe_2_Cluster1_pt1.sh;
bash mCAD_exe_3_Cluster1_pt1.sh;
bash mCAD_exe_4_Cluster1_pt1.sh;
bash mCAD_exe_5_Cluster1_pt1.sh;
python3 include/postProcess.py

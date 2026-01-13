echo "[Resampling] Test start!"
echo ">>> Canonical Capsules >>>"
python test_orientation.py --align_caca_ckpt ckpts/multi_caca/best_model.pth --label_in_use 0,3,4,10 --resample
echo "<<< Canonical Capsules <<<"
echo ">>> ConDor >>>"
python test_orientation.py --align_condor_ckpt ckpts/multi_condor.ckpt --label_in_use 0,3,4,10 --resample
echo "<<< ConDor <<<"
echo ">>> VN-SPD >>>"
python test_orientation.py --align_vnspd_dir ckpts/multi_vnspd --label_in_use 0,3,4,10 --resample
echo "<<< VN-SPD <<<"
echo ">>> Ours >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_ours-v2/best.ckpt --label_in_use 0,3,4,10 --resample
echo "<<< Ours <<<"
echo ">>> Point Transformer >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_pt/best.ckpt --label_in_use 0,3,4,10 --resample
echo "<<< Point Transformer <<<"
echo "[Resampling] Test end!"

echo ""
echo ""
echo "[Noise] Test start!"
echo ">>> Canonical Capsules >>>"
python test_orientation.py --align_caca_ckpt ckpts/multi_caca/best_model.pth --label_in_use 0,3,4,10 --add_gaussian
echo "<<< Canonical Capsules <<<"
echo ">>> ConDor >>>"
python test_orientation.py --align_condor_ckpt ckpts/multi_condor.ckpt --label_in_use 0,3,4,10 --add_gaussian
echo "<<< ConDor <<<"
echo ">>> VN-SPD >>>"
python test_orientation.py --align_vnspd_dir ckpts/multi_vnspd --label_in_use 0,3,4,10 --add_gaussian
echo "<<< VN-SPD <<<"
echo ">>> Ours >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_ours-v2/best.ckpt --label_in_use 0,3,4,10 --add_gaussian
echo "<<< Ours <<<"
echo ">>> Point Transformer >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_pt/best.ckpt --label_in_use 0,3,4,10 --add_gaussian
echo "<<< Point Transformer <<<"
echo "[Noise] Test end!"

echo ""
echo ""
echo "[KnnPatchRemoval] Test start!"
echo ">>> Canonical Capsules >>>"
python test_orientation.py --align_caca_ckpt ckpts/multi_caca/best_model.pth --label_in_use 0,3,4,10 --knn_removal
echo "<<< Canonical Capsules <<<"
echo ">>> ConDor >>>"
python test_orientation.py --align_condor_ckpt ckpts/multi_condor.ckpt --label_in_use 0,3,4,10 --knn_removal
echo "<<< ConDor <<<"
echo ">>> VN-SPD >>>"
python test_orientation.py --align_vnspd_dir ckpts/multi_vnspd --label_in_use 0,3,4,10 --knn_removal
echo "<<< VN-SPD <<<"
echo ">>> Ours >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_ours-v2/best.ckpt --label_in_use 0,3,4,10 --knn_removal
echo "<<< Ours <<<"
echo ">>> Point Transformer >>>"
python test_orientation.py --align_pl_ckpt ckpts/multi_pt/best.ckpt --label_in_use 0,3,4,10 --knn_removal
echo "<<< Point Transformer <<<"
echo "[KnnPatchRemoval] Test end!"
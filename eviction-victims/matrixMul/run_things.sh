export NAME="matMult"
export TRACE_KERNEL="MatrixMulCUDA"
export TRACE_INSTANCE=1

export APP_NAME="matrixMul"

echo "Collecting traces with NVBit..."
# LD_PRELOAD=../../NVBitTest/mem_traceV2/mem_traceV2.so ./iter < partition | sudo tee ${NAME}Trace > /dev/null
LD_PRELOAD=../../NVBitTest/mem_traceV2/mem_traceV2.so ./${APP_NAME}  | sudo tee ${NAME}Trace > /dev/null

echo "Formatting traces..."
sudo traceFormat < ${NAME}Trace > ${NAME}Fmt 2> ${NAME}VirtBase
VIRT_BASE=$(cat ${NAME}VirtBase)
echo "Formating completed. Base virt addr is $VIRT_BASE"
echo "Sorting formatted trace..."
sudo cut -d " " -f 2,4,6,8,12 ${NAME}Fmt | sudo sort -n -s -k 1,1 | sudo tee ${NAME}SortedFmt > /dev/null


echo "Simulating reuses..."
sudo cut -d " " -f 2,3,4,5 ${NAME}SortedFmt | traceReuse -o ${VIRT_BASE} 20 10 1 > ${NAME}Reuse

echo "Simulating L2 cache..."
sudo cut -d " " -f 5 ${NAME}SortedFmt | L2Sim -o ${VIRT_BASE} > ${NAME}L2Sim 2> ${NAME}L2Stat
# sudo cut -d " " -f 2,3,4,5 ${NAME}SortedFmt | traceBankReuse -o ${VIRT_BASE} > ${NAME}BankReuse

echo "Sorting reuse trace..."
sudo cut -d " " -f 1,3,5,7,13 ${NAME}Reuse | sudo sort -n -s -k 5,5 | sudo tee ${NAME}SortedReuse > /dev/null
# sudo cut -d " " -f 1,3,5,7,13 ${NAME}BankReuse | sudo sort -n -s -k 5,5 | sudo tee ${NAME}SortedBankReuse > /dev/null

echo "Collecting line statistics..."
sudo lineCounts <${NAME}SortedReuse > ${NAME}LineCounts

echo "Collecting warp info..."
sudo python3 ./warpInfo.py ${NAME}SortedReuse 20 10 1 > ${NAME}WarpInfo 2> ${NAME}MemInfo
# sudo bankCounts <${NAME}SortedBankReuse > ${NAME}BankCounts
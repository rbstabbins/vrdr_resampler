#!/bin/bash
# ==========================================
# CRISM VRDR hybrid downloader (FTP listing → HTTPS download)
# Stable version — downloads all mcXX folders under /Volumes/[volume name]/vrdr/
# ==========================================

DEST="/Volumes/[volume name]/vrdr/"
FTP_ROOT="/mro/mro-m-crism-5-rdr-vnirhyperspectral-v1/mrocr_7001/vrdr"
BASE_HTTPS="https://pds-geosciences.wustl.edu"
LOGDIR="$DEST/CRISM_logs"
PARALLEL_JOBS=3

# All mosaic folders
FOLDERS=(mc16 mc23)

mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/CRISM_VRDR_all_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$MASTER_LOG"
echo "CRISM VRDR hybrid HTTPS download started at $(date)" | tee -a "$MASTER_LOG"
echo "Destination: $DEST" | tee -a "$MASTER_LOG"
echo "Parallel jobs per folder: $PARALLEL_JOBS" | tee -a "$MASTER_LOG"
echo "Log directory: $LOGDIR" | tee -a "$MASTER_LOG"
echo "==========================================" | tee -a "$MASTER_LOG"

for cmd in lftp wget parallel; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Please install $cmd (brew install $cmd)"; exit 1; }
done

# Color helpers (optional)
green=$(tput setaf 2); yellow=$(tput setaf 3); reset=$(tput sgr0)

for F in "${FOLDERS[@]}"; do
  echo "------------------------------------------" | tee -a "$MASTER_LOG"
  echo "${yellow}Processing folder: $F${reset}" | tee -a "$MASTER_LOG"

  TMP_LIST="/tmp/vrral_${F}_list.txt"
  HTTPS_LIST="/tmp/vrral_${F}_https.txt"
  FOLDER_LOG="$LOGDIR/${F}_$(date +%Y%m%d_%H%M%S).log"

  # Step 1: Get file list from FTP
  lftp -u anonymous,anonymous ftp://pds-geosciences.wustl.edu <<EOF > "$TMP_LIST"
set ftp:list-options -a
set ftp:use-feat no
set ftp:ssl-allow no
set ftp:passive-mode yes
find $FTP_ROOT/$F
exit
EOF

  grep -i 'vrral' "$TMP_LIST" | grep -E '\.(img|hdr|lbl|xml|tab)$' > "${TMP_LIST}.filtered" || true
  mv "${TMP_LIST}.filtered" "$TMP_LIST"

  COUNT=$(wc -l < "$TMP_LIST" | tr -d ' ')
  echo "Found $COUNT vrral files in $F" | tee -a "$MASTER_LOG"

  if [ "$COUNT" -eq 0 ]; then
    echo "No files found for $F; skipping." | tee -a "$MASTER_LOG"
    continue
  fi

  # Step 2: Convert FTP path → HTTPS URL
  sed "s|^$FTP_ROOT|$BASE_HTTPS$FTP_ROOT|" "$TMP_LIST" > "$HTTPS_LIST"

  echo "Sample URLs:" | tee -a "$MASTER_LOG"
  head -n 3 "$HTTPS_LIST" | tee -a "$MASTER_LOG"

  # Step 3: Download each via HTTPS, fixing local path
  echo "${green}Starting parallel download for $F...${reset}" | tee -a "$MASTER_LOG"
  cat "$HTTPS_LIST" | parallel -j "$PARALLEL_JOBS" --bar '
    URL={}
    RELATIVE_PATH=$(echo "$URL" | sed "s|^'"$BASE_HTTPS$FTP_ROOT"'||")
    LOCAL_PATH="'"$DEST"'$RELATIVE_PATH"
    mkdir -p "$(dirname "$LOCAL_PATH")"
    if [ -f "$LOCAL_PATH" ]; then
      echo "Skipping existing $(basename "$LOCAL_PATH")" >>"'"$FOLDER_LOG"'"
    else
      wget -c -q --show-progress "$URL" -O "$LOCAL_PATH" && \
      echo "Downloaded: $URL" >>"'"$FOLDER_LOG"'"
    fi
  '

  echo "Completed folder: $F at $(date)" | tee -a "$MASTER_LOG"
done

echo "=========================================="
echo "${green}All folders completed at $(date)${reset}"
echo "=========================================="

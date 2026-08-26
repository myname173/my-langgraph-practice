$base='c:/Users/13682/Desktop/my-langgraph-practice-main'
$targets=@("$base/data/multimedia_checkpoints.sqlite","$base/data/multimedia_checkpoints.sqlite-wal","$base/data/multimedia_checkpoints.sqlite-shm","$base/workspace/_make_dev.out.log","$base/workspace/_make_dev.err.log")
$deadline=(Get-Date).AddMinutes(12)
$result="PENDING"
while((Get-Date) -lt $deadline){
  $allGone=$true
  foreach($t in $targets){ if(Test-Path $t){ try{ Remove-Item $t -Force -ErrorAction Stop }catch{ $allGone=$false } } }
  if($allGone){ $result="SUCCESS_ALL_GONE"; break }
  Start-Sleep -Seconds 20
}
($result) | Out-File -Encoding utf8 "$base/workspace/_del_result.txt"

$job = Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 40; $i++) {
        try {
            $r = Invoke-WebRequest -Uri http://localhost:2024/ok -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { "READY after $($i)s"; break }
        } catch {}
        Start-Sleep -Seconds 1
    }
}
$wait = Wait-Job -Job $job -Timeout 50
if ($wait.State -eq 'Completed') { Receive-Job -Job $job } else { "SVC NOT READY in 50s" }

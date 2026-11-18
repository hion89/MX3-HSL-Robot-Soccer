$ErrorActionPreference = 'Stop'
$bundle = Join-Path $PSScriptRoot 'kick22300_20261118'
$server = 'bami122@163.180.145.122'
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'manifest.json'))) {
    throw "Missing transfer bundle: $bundle"
}
$manifest = Get-Content -LiteralPath (Join-Path $bundle 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $bundle $entry.Name)).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "Local checksum mismatch: $($entry.Name)" }
}
ssh $server 'mkdir -p /home/bami122/incoming'
if ($LASTEXITCODE -ne 0) { throw 'SSH connection or incoming creation failed; no payload sent.' }
scp -r $bundle "${server}:/home/bami122/incoming/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; installation not executed.' }
ssh $server 'python3 /home/bami122/incoming/kick22300_20261118/install.py'
if ($LASTEXITCODE -ne 0) { throw 'Remote installation failed; inspect the message above.' }
Write-Host 'TRANSFER_FINISHED: v1_5b 22300 imported into v1_6c log directory.'

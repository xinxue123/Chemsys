$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$legacyRoot = Join-Path $workspaceRoot 'chemist_system'
$archiveRoot = Join-Path $workspaceRoot 'experiments\chemist_system'
$sourceRoot = Join-Path $workspaceRoot 'code\chemist_system'
$metaRoot = Join-Path $workspaceRoot 'runtime\layout_20261030'

function Assert-WorkspacePath([string]$candidate) {
    $full = [IO.Path]::GetFullPath($candidate)
    if (-not $full.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside workspace: $full"
    }
    return $full
}
function Ensure-Parent([string]$path) {
    $parent = Split-Path -Parent (Assert-WorkspacePath $path)
    [IO.Directory]::CreateDirectory($parent) | Out-Null
}
function Save-Json([string]$path, $value) {
    Ensure-Parent $path
    [IO.File]::WriteAllText($path, ($value | ConvertTo-Json -Depth 12) + "`n", [Text.UTF8Encoding]::new($false))
}
function Move-Checked([string]$source, [string]$target) {
    $source = Assert-WorkspacePath $source
    $target = Assert-WorkspacePath $target
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Target already exists: $target" }
    # Every directory move has a verified, workspace-contained absolute source
    # and destination. File operations stay in native PowerShell throughout.
    Ensure-Parent $target
    if (Test-Path -LiteralPath $source -PathType Container) {
        # Rename the directory entry without traversing contained junctions.
        [IO.Directory]::Move($source, $target)
    } else {
        Move-Item -LiteralPath $source -Destination $target
    }
}
function Move-With-Junction([string]$relative, [string]$targetRelative) {
    $source = Join-Path $legacyRoot $relative
    if (-not (Test-Path -LiteralPath $source)) { return }
    $target = Join-Path $workspaceRoot $targetRelative
    Move-Checked $source $target
    New-Item -ItemType Junction -Path $source -Target $target | Out-Null
    $script:directoryMoves.Add(@{original_relative=$relative;canonical_relative=$targetRelative;compatibility='junction'})
}

if (Test-Path -LiteralPath (Join-Path $metaRoot 'migration_manifest.json')) {
    throw 'Migration already planned. Inspect its journal before repeating file operations.'
}
if (-not (Test-Path -LiteralPath $legacyRoot -PathType Container)) { throw 'Original project not found.' }
if ((Get-Item -LiteralPath $legacyRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Original project is already a compatibility link.'
}
if (Test-Path -LiteralPath $archiveRoot) { throw 'Experiment destination already exists.' }

$items = [Collections.Generic.List[object]]::new()
foreach ($folder in @('chem','compat','scripts','tests','third_party')) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $legacyRoot $folder) -Recurse -File) {
        if ($file.FullName -match '\\(__pycache__|node_modules|\.git|\.pytest_cache)\\' -or $file.Extension -in @('.pyc','.pyo')) { continue }
        $relative = $file.FullName.Substring($legacyRoot.Length + 1)
        $items.Add(@{legacy_relative=$relative.Replace('\','/');canonical_relative=('code/chemist_system/' + $relative.Replace('\','/'));sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
    }
}
foreach ($file in Get-ChildItem -LiteralPath $legacyRoot -File) {
    if ($file.Extension -ne '.py' -and $file.Name -notin @('pyproject.toml','README.md','.gitignore')) { continue }
    $items.Add(@{legacy_relative=$file.Name;canonical_relative=('code/chemist_system/' + $file.Name);sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $legacyRoot 'research') -Recurse -File) {
    if ($file.FullName -match '\\(node_modules|__pycache__|\.git)\\' -or $file.Extension -notin @('.py','.sh','.ps1','.bat','.ipynb')) { continue }
    $relative = $file.FullName.Substring($legacyRoot.Length + 1).Replace('\','/')
    $items.Add(@{legacy_relative=$relative;canonical_relative=('code/experiment_tools/' + $relative);sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
}
$checkScript = Join-Path $legacyRoot 'outputs\suzuki_debug_viewer_check.js'
if (Test-Path -LiteralPath $checkScript) {
    $items.Add(@{legacy_relative='outputs/suzuki_debug_viewer_check.js';canonical_relative='code/experiment_tools/outputs/suzuki_debug_viewer_check.js';sha256=(Get-FileHash -LiteralPath $checkScript -Algorithm SHA256).Hash.ToLowerInvariant()})
}
$plan = @{layout_version=1;workspace=$workspaceRoot;legacy_path='chemist_system';experiments_path='experiments/chemist_system';
    code_path='code/chemist_system';created_utc=[DateTime]::UtcNow.ToString('o');status='PLANNED';
    source_files=@($items.ToArray());scientific_content_policy='byte-identical relocation; old names retained as hard links; no algorithm or experiment changes'}
Save-Json (Join-Path $metaRoot 'migration_manifest.json') $plan
Write-Output "Planned source relocation: $($items.Count) files"

$done = 0
foreach ($item in $items) {
    $source = Assert-WorkspacePath (Join-Path $legacyRoot $item.legacy_relative)
    $target = Assert-WorkspacePath (Join-Path $workspaceRoot $item.canonical_relative)
    Move-Checked $source $target
    # NTFS compatibility names share the same bytes; no duplicate source copy.
    New-Item -ItemType HardLink -Path $source -Target $target | Out-Null
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw "Source hash mismatch: $target" }
    $done += 1
    if ($done % 100 -eq 0) { Write-Output "Source files relocated and linked: $done" }
}
$directoryMoves = [Collections.Generic.List[object]]::new()
foreach ($envDirectory in Get-ChildItem -LiteralPath $legacyRoot -Directory -Force | Where-Object {$_.Name -like '.venv*'}) {
    Move-With-Junction $envDirectory.Name ('runtime/environments/' + $envDirectory.Name)
}
Move-With-Junction 'docs' 'docs/chemist_system'
Move-With-Junction '.pytest_cache' 'runtime/cache/pytest'
Move-With-Junction 'outputs/tex_cache' 'runtime/cache/tex'
Move-With-Junction 'outputs/tex_tools' 'runtime/tools/tex_tools'
Move-With-Junction 'outputs/rdkit_runtime' 'runtime/tools/rdkit_runtime'
Move-With-Junction 'research/suzuki_case2_csv_export/node_modules' 'runtime/node_modules/suzuki_case2_csv_export'

Move-Checked $legacyRoot $archiveRoot
New-Item -ItemType Junction -Path $legacyRoot -Target $archiveRoot | Out-Null
$plan.status='MOVED_PENDING_VERIFICATION'
$plan.directory_moves=@($directoryMoves.ToArray())
$plan.source_files_moved=$done
$plan.completed_utc=[DateTime]::UtcNow.ToString('o')
Save-Json (Join-Path $metaRoot 'migration_manifest.json') $plan
Write-Output "Layout migrated. Code: $sourceRoot"
Write-Output "Experiments: $archiveRoot"
Write-Output 'Original paths remain accessible. Next run code/tools/verify_layout.py.'

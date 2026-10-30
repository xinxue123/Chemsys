$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$originalRoot = Join-Path $workspaceRoot 'chemist_system'
$archiveRoot = Join-Path $workspaceRoot 'experiments\chemist_system'
$metaRoot = Join-Path $workspaceRoot 'runtime\layout_20261030'
$manifestPath = Join-Path $metaRoot 'migration_manifest.json'
$plan = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$script:directoryRenames = 0
$script:fileRenames = 0
$script:duplicateBackups = 0

function Checked([string]$candidate) {
    $full = [IO.Path]::GetFullPath($candidate)
    if (-not $full.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $full" }
    return $full
}
function Is-Junction([string]$path) {
    return [bool]((Get-Item -LiteralPath (Checked $path) -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)
}
function Junction-Target([string]$path) {
    return Checked ([string](Get-Item -LiteralPath (Checked $path) -Force).Target)
}
function Make-Junction([string]$path,[string]$target) {
    $path=Checked $path;$target=Checked $target
    [IO.Directory]::CreateDirectory((Split-Path -Parent $path)) | Out-Null
    New-Item -ItemType Junction -Path $path -Target $target | Out-Null
}
function Merge-Without-Following-Links([string]$source,[string]$target) {
    $source=Checked $source;$target=Checked $target
    if (-not (Test-Path -LiteralPath $source)) { return }
    if (-not (Test-Path -LiteralPath $target)) {
        [IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
        [IO.Directory]::Move($source,$target)
        $script:directoryRenames += 1
        return
    }
    if (Is-Junction $source) {
        if (-not (Is-Junction $target) -or (Junction-Target $source) -ne (Junction-Target $target)) { throw "Ambiguous junction: $source" }
        # This removes one verified junction name only; it never traverses it.
        [IO.Directory]::Delete($source,$false)
        return
    }
    if (Is-Junction $target) { throw "Refusing to merge through a junction: $target" }
    foreach ($entry in [IO.Directory]::EnumerateFileSystemEntries($source)) {
        $entry=Checked $entry
        $destination=Checked (Join-Path $target ([IO.Path]::GetFileName($entry)))
        if (Test-Path -LiteralPath $entry -PathType Container) {
            Merge-Without-Following-Links $entry $destination
        } elseif (Test-Path -LiteralPath $destination) {
            if ((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) { throw "Different overlapping files: $entry" }
            $backup=Checked (Join-Path $metaRoot ('overlap_backups\'+$entry.Substring($workspaceRoot.Length+1)))
            [IO.Directory]::CreateDirectory((Split-Path -Parent $backup)) | Out-Null
            [IO.File]::Move($entry,$backup)
            $script:duplicateBackups += 1
        } else {
            [IO.File]::Move($entry,$destination)
            $script:fileRenames += 1
        }
    }
    if (@([IO.Directory]::EnumerateFileSystemEntries($source)).Count -ne 0) { throw "Source not empty: $source" }
    [IO.Directory]::Delete($source,$false)
}

if ($plan.status -ne 'PLANNED') { throw 'Repair is only for the interrupted planned migration.' }
$moves = @(
    @{original_relative='.venv-baselines';canonical_relative='runtime/environments/.venv-baselines'},
    @{original_relative='.venv-official';canonical_relative='runtime/environments/.venv-official'},
    @{original_relative='.venv-reproduce';canonical_relative='runtime/environments/.venv-reproduce'},
    @{original_relative='.venv-research';canonical_relative='runtime/environments/.venv-research'},
    @{original_relative='.venv-summit';canonical_relative='runtime/environments/.venv-summit'},
    @{original_relative='docs';canonical_relative='docs/chemist_system'},
    @{original_relative='.pytest_cache';canonical_relative='runtime/cache/pytest'},
    @{original_relative='outputs/tex_cache';canonical_relative='runtime/cache/tex'},
    @{original_relative='outputs/tex_tools';canonical_relative='runtime/tools/tex_tools'},
    @{original_relative='outputs/rdkit_runtime';canonical_relative='runtime/tools/rdkit_runtime'},
    @{original_relative='research/suzuki_case2_csv_export/node_modules';canonical_relative='runtime/node_modules/suzuki_case2_csv_export'}
)
foreach ($move in $moves) {
    $partial=Checked (Join-Path $archiveRoot $move.original_relative)
    $canonical=Checked (Join-Path $workspaceRoot $move.canonical_relative)
    $original=Checked (Join-Path $originalRoot $move.original_relative)
    if ((Test-Path -LiteralPath $partial) -and -not (Is-Junction $partial)) {
        Merge-Without-Following-Links $partial $canonical
        Make-Junction $partial $canonical
    } elseif (-not (Test-Path -LiteralPath $partial)) {
        if (-not (Test-Path -LiteralPath $canonical)) { throw "Missing canonical directory: $canonical" }
        Make-Junction $partial $canonical
    }
    if ((Junction-Target $partial) -ne $canonical) { throw "Wrong target: $partial" }
    $move.compatibility='junction'
    Write-Output "Directory verified: $($move.canonical_relative)"
}
Merge-Without-Following-Links $originalRoot $archiveRoot
Make-Junction $originalRoot $archiveRoot
foreach ($move in $moves | Where-Object {$_.original_relative -like '.venv*'}) {
    if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $workspaceRoot $move.canonical_relative) 'pyvenv.cfg'))) { throw 'Incomplete virtual environment' }
}
$plan.status='MOVED_PENDING_VERIFICATION'
$plan | Add-Member -NotePropertyName directory_moves -NotePropertyValue $moves -Force
$plan | Add-Member -NotePropertyName source_files_moved -NotePropertyValue $plan.source_files.Count -Force
$plan | Add-Member -NotePropertyName completed_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
[IO.File]::WriteAllText($manifestPath,($plan | ConvertTo-Json -Depth 12)+"`n",[Text.UTF8Encoding]::new($false))
$receipt=@{status='COMPLETED_WITHOUT_RECURSIVE_LINK_TRAVERSAL';directory_renames=$script:directoryRenames;
    file_renames=$script:fileRenames;identical_overlap_backups=$script:duplicateBackups;completed_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $metaRoot 'move_repair.json'),($receipt | ConvertTo-Json)+"`n",[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json

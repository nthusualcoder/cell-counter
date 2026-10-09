# 수정한 코드를 GitHub에 올려서 다른 PC가 [업데이트 확인] 버튼으로 받을 수 있게 합니다.
#   1) VERSION 끝자리를 1 올림  2) 모든 변경 사항 commit  3) GitHub(main 브랜치)에 push
# 실행: 저장소 폴더의 publish_update.bat 더블클릭
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path (Join-Path $PWD '.git'))) {
    Write-Host '이 폴더는 개발용 git 저장소가 아닙니다. 저장소 폴더(Cellpose)의 publish_update.bat 을 실행하세요.'
    exit 1
}

git status --short
$msg = Read-Host '변경 내용을 한 줄로 입력하세요 (취소: 빈 칸으로 엔터)'
if ([string]::IsNullOrWhiteSpace($msg)) { Write-Host '취소했습니다.'; exit 1 }

$versionFile = Join-Path $PWD 'VERSION'
$old = Get-Content $versionFile -Raw
$v = $old.Trim().Split('.')
$v[2] = [string]([int]$v[2] + 1)
$new = $v -join '.'
[IO.File]::WriteAllText($versionFile, "$new`n")

function Fail($text) {
    [IO.File]::WriteAllText($versionFile, $old)  # 버전 번호를 원래대로 되돌림
    Write-Host $text
    exit 1
}

git add -A
if ($LASTEXITCODE -ne 0) { Fail 'git add 실패: 위 메시지를 확인하세요.' }
$tmp = [IO.Path]::GetTempFileName()
[IO.File]::WriteAllText($tmp, "v${new}: $msg", (New-Object Text.UTF8Encoding $false))
git commit -F $tmp
$rc = $LASTEXITCODE
Remove-Item $tmp
if ($rc -ne 0) { git reset -q; Fail '커밋 실패: 위 git 메시지를 확인하세요.' }

git push -u origin HEAD:main
if ($LASTEXITCODE -ne 0) { Write-Host 'push 실패: 인터넷/GitHub 로그인 상태를 확인한 뒤 다시 실행하세요 (커밋은 저장되어 있음).'; exit 1 }
Write-Host "v$new 업로드 완료. 다른 PC에서 [업데이트 확인]을 누르면 반영됩니다 (GitHub 반영까지 몇 분 걸릴 수 있음)."

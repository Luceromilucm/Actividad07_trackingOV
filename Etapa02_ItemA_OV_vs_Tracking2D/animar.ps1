# Lee base.pptx, inserta los videos y agrega las animaciones segun el nombre de cada forma
# (ver build_deck.py). Guarda ..\etapa02-actividad1.pptx y exporta PNG de revision.
# Uso: powershell -ExecutionPolicy Bypass -File animar.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $here "base.pptx"
$out  = Join-Path (Split-Path -Parent $here) "etapa02-actividad1.pptx"
$vids = Join-Path $here "videos"
$png  = Join-Path $here "revision"
New-Item -ItemType Directory -Force $png | Out-Null

$effects  = @{ fade = 10; wipe = 22; zoom = 53; fly = 2; appear = 1 }
$triggers = @{ click = 1; with = 2; after = 3 }
$PLAY = 83   # msoAnimEffectMediaPlay

$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($src, $false, $false, $false)

foreach ($sl in $pres.Slides) {
    $items = @()
    foreach ($sh in $sl.Shapes) {
        if ($sh.Name -match '^([AV])(\d{2})_(.+)_(click|with|after)$') {
            $items += [pscustomobject]@{ Shape = $sh; Kind = $Matches[1]; Order = [int]$Matches[2];
                                         Arg = $Matches[3]; Trig = $Matches[4]; Z = $sh.ZOrderPosition }
        }
    }
    $items = $items | Sort-Object Order, @{ Expression = { if ($_.Kind -eq 'A') { 0 } else { 1 } } }, Z
    $seq = $sl.TimeLine.MainSequence
    $prevOrder = -1
    foreach ($it in $items) {
        $trig = $triggers[$it.Trig]
        if ($it.Order -eq $prevOrder) { $trig = 2 }
        $prevOrder = $it.Order
        if ($it.Kind -eq 'V') {
            $ph = $it.Shape
            $file = Join-Path $vids ($it.Arg + ".mp4")
            $v = $sl.Shapes.AddMediaObject2($file, 0, -1, $ph.Left, $ph.Top, $ph.Width, $ph.Height)
            $v.Name = "video_" + $it.Arg
            $ph.Delete()
            # quitar efectos que PowerPoint haya puesto solo
            for ($k = $seq.Count; $k -ge 1; $k--) { if ($seq.Item($k).Shape.Name -eq $v.Name) { $seq.Item($k).Delete() } }
            $e = $seq.AddEffect($v, $PLAY, 0, $trig)
            Write-Host ("  slide {0}: video {1} ({2})" -f $sl.SlideIndex, $it.Arg, $it.Trig)
        } else {
            $id = $effects[$it.Arg]
            try { $e = $seq.AddEffect($it.Shape, $id, 0, $trig) }
            catch { $e = $seq.AddEffect($it.Shape, 10, 0, $trig) }
            $e.Timing.Duration = 0.5
            if ($it.Arg -eq 'wipe') { try { $e.EffectParameters.Direction = 4 } catch {} }
            if ($trig -eq 3) { $e.Timing.TriggerDelayTime = 0.08; $e.Timing.Duration = 0.35 }
            $it.Shape.Name = $it.Shape.Name -replace '^[AV]\d{2}_', 'anim_'
        }
    }
    $sl.SlideShowTransition.EntryEffect = 3849   # ppEffectFadeSmoothly
    $sl.SlideShowTransition.Duration = 0.6
    $sl.Export((Join-Path $png ("slide{0:00}.png" -f $sl.SlideIndex)), "PNG", 1600, 900)
}

$pres.SaveAs($out)
$pres.Close()
$pp.Quit()
Write-Host "Guardado: $out"

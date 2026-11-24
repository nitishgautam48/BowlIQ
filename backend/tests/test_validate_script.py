from app.cli.validate import main

RESULT = {
    "video": {"fps": 120.0},
    "events": {"bfc": {"frame": 40}, "ffc": {"frame": 58}, "release": {"frame": 67}},
    "metrics": [{"key": "front_knee_angle_ffc", "value": 160.0, "reason": None},
                {"key": "stride_length_pct_height", "value": 80.0, "reason": None}],
}


def test_validate_prints_mae_and_frame_error(tmp_path, capsys):
    runs = tmp_path / "runs.csv"
    runs.write_text("clip,analysis_id,status,error\nnets1.mp4,abc,done,\n")
    manual = tmp_path / "manual.csv"
    manual.write_text("clip,metric,manual_value,event,manual_frame\n"
                      "nets1.mp4,front_knee_angle_ffc,156,,\n"
                      "nets1.mp4,stride_length_pct_height,83,,\n"
                      "nets1.mp4,,,FFC,57\n"
                      "nets1.mp4,,,release,67\n"
                      "missing.mp4,,,FFC,1\n")
    assert main([str(manual), "--runs", str(runs)], load=lambda aid: RESULT if aid == "abc" else None) == 0
    out = capsys.readouterr().out
    assert "front_knee_angle_ffc" in out and "4.0" in out
    assert "stride_length_pct_height" in out and "3.0" in out
    assert "FFC" in out and "release" in out
    assert "missing.mp4: no analysis found" in out

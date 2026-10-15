import os
import unittest

import numpy as np

from vigilia.config import Settings, ConfigError, validate_camera
from vigilia.decision import decide
from vigilia.heuristics import TrackState, analyze_pose, update_track
from vigilia.vision import Person


def kp(over=None):
    """Persona sentada y erguida (coordenadas de 640x360)."""
    pts = {0: (320, 100), 1: (310, 92), 2: (330, 92), 3: (298, 96), 4: (342, 96), 5: (280, 150), 6: (360, 150),
           7: (270, 200), 8: (370, 200), 9: (290, 240), 10: (350, 240), 11: (290, 260), 12: (350, 260)}
    pts.update(over or {})
    k = np.zeros((17, 3))
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.9)
    return k


BBOX = np.array([250, 70, 390, 300.0])


class CameraValidation(unittest.TestCase):
    def test_url_is_split_and_password_masked(self):
        cam = validate_camera({"url": "rtsp://admin:p%40ss@192.168.1.10:554/Streaming/Channels/102"})
        self.assertEqual((cam.user, cam.password, cam.host, cam.port), ("admin", "p@ss", "192.168.1.10", 554))
        self.assertIn("p%40ss", cam.source())
        self.assertNotIn("p%40ss", cam.safe_url())
        self.assertNotIn("password", cam.public())

    def test_url_without_credentials_keeps_separate_user_password(self):
        cam = validate_camera({"url": "rtsp://192.168.1.3:8080/h264_ulaw.sdp", "user": "avalos", "password": "x"})
        self.assertEqual((cam.user, cam.password), ("avalos", "x"))
        self.assertEqual(cam.path, "/h264_ulaw.sdp")

    def test_brand_default_path_and_port(self):
        cam = validate_camera({"host": "10.0.0.5", "brand": "dahua", "user": "u", "password": "x"})
        self.assertEqual(cam.port, 554)
        self.assertIn("subtype=1", cam.path)

    def test_rejects_bad_input(self):
        for bad in ({"host": "a/b"}, {"host": "10.0.0.5", "port": 99999}, {"host": "10.0.0.5", "protocol": "file"},
                    {"host": "10.0.0.5", "id": "../x"}, {"host": "10.0.0.5", "password": "x"},
                    {"host": "10.0.0.5", "user": "a\nb"}, {"host": "10.0.0.5", "path": "rtsp://evil"},
                    {"host": "10.0.0.5", "roi": [[0, 0], [2, 2], [1, 1]]}, {"host": ""}):
            with self.assertRaises(ValueError, msg=bad):
                validate_camera(bad)

    def test_existing_password_kept_on_edit(self):
        cam = validate_camera({"host": "10.0.0.5", "user": "u"}, existing_password="secreto")
        self.assertEqual(cam.password, "secreto")


class SettingsValidation(unittest.TestCase):
    def test_invalid_values_reported_together(self):
        os.environ.update(INFERENCE_FPS="abc", YOLO_CONF="5", WEB_HOST="0.0.0.0")
        try:
            with self.assertRaises(ConfigError) as ctx:
                Settings.from_env()
            msg = str(ctx.exception)
            self.assertIn("INFERENCE_FPS", msg)
            self.assertIn("YOLO_CONF", msg)
            self.assertIn("DASHBOARD_PASSWORD", msg)
        finally:
            for k in ("INFERENCE_FPS", "YOLO_CONF", "WEB_HOST"):
                os.environ.pop(k, None)

    def test_defaults_ok(self):
        self.assertEqual(Settings.from_env().inference_fps, 5.0)


class Posture(unittest.TestCase):
    def test_upright(self):
        m = analyze_pose(kp(), BBOX)
        self.assertEqual(m.posture, "erguido")
        self.assertLess(m.posture_score, 0.2)

    def test_head_down(self):
        m = analyze_pose(kp({0: (320, 160), 1: (312, 154), 2: (328, 154), 3: (300, 150), 4: (340, 150)}), BBOX)
        self.assertEqual(m.posture, "cabeza_baja")
        self.assertGreater(m.posture_score, 0.45)

    def test_lying_by_aspect(self):
        m = analyze_pose(kp(), np.array([0, 100, 400, 220.0]))
        self.assertEqual(m.posture, "recostado")

    def test_head_hidden(self):
        k = kp()
        k[:5, 2] = 0.0
        m = analyze_pose(k, BBOX)
        self.assertTrue(m.head_hidden)
        self.assertEqual(m.posture, "sin_cabeza")


class Escalation(unittest.TestCase):
    def run_track(self, kpts, seconds, step=0.2):
        ts = TrackState(1, "C", 0.0, 0.0)
        s = Settings.from_env()
        t = 0.0
        while t < seconds:
            update_track(ts, Person(1, BBOX, 0.9, kpts), None, t, s.suspect_seconds, s.sleep_seconds)
            t += step
        return ts, s, t

    def test_upright_never_suspect(self):
        ts, s, now = self.run_track(kp(), 120)
        self.assertEqual(ts.local_state, "despierta")
        self.assertEqual(decide(ts, now, s)[0], "ok")

    def test_sustained_head_down_goes_local_asleep_then_alerts_without_gemini(self):
        down = kp({0: (320, 170), 1: (312, 164), 2: (328, 164)})
        ts, s, now = self.run_track(down, 20)
        self.assertEqual(ts.local_state, "somnolienta")
        self.assertEqual(decide(ts, now, s)[0], "ok")           # aun no: respaldo local exige mas tiempo
        ts, s, now = self.run_track(down, 60)
        self.assertIn(decide(ts, now, s)[0], ("somnolencia", "dormido"))

    def test_gemini_awake_vetoes_and_asleep_escalates(self):
        down = kp({0: (320, 170), 1: (312, 164), 2: (328, 164)})
        ts, s, now = self.run_track(down, 20)
        ts.verdict = {"state": "dormida", "confidence": 0.9, "reasoning": "ojos cerrados"}
        ts.verdict_time = now
        self.assertEqual(decide(ts, now, s)[0], "dormido")
        ts.verdict = {"state": "despierta", "confidence": 0.9, "reasoning": "usa celular"}
        ts.veto_until = now + 60
        self.assertEqual(decide(ts, now, s)[0], "ok")

    def test_stale_verdict_ignored_when_person_recovers(self):
        ts, s, now = self.run_track(kp(), 30)
        ts.verdict = {"state": "dormida", "confidence": 0.95, "reasoning": "x"}
        ts.verdict_time = now
        self.assertEqual(decide(ts, now, s)[0], "ok")


if __name__ == "__main__":
    unittest.main()

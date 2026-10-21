"""
FACESCAN AI bridge server.

Adds an HTTP API on top of the existing attendance system WITHOUT changing the
recognition approach: same face_recognition calls, same 0.25 frame scale, same
compare_faces/face_distance matching, same Attendance.csv format (Name,Time,Date).
AttendanceProject.py cannot be imported (it starts its webcam loop at import
time), so this file runs the one recognition loop itself. Do not run both at once.
Face encodings never leave this process.
"""
import csv, logging, os, sys, threading, time
from collections import defaultdict
from datetime import datetime
from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('facescan')
PORT = int(os.environ.get('PORT', 5001))

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(BASE, 'Images_Attendance')
CSV_PATH = os.path.join(BASE, 'Attendance.csv')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
lock = threading.RLock()
app = Flask(__name__)


def roster():
    """Enrolled people = image filenames (the only student data that exists)."""
    if not os.path.isdir(IMAGES):
        return None
    return sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.lower().endswith(IMG_EXT))


def read_records():
    if not os.path.exists(CSV_PATH):
        return []
    out = []
    with lock, open(CSV_PATH, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0].strip().lower() == 'name':
                continue  # blank lines and the header
            try:
                d = datetime.strptime(row[2].strip(), '%d/%m/%Y').date()
            except ValueError:
                continue
            out.append({'name': row[0].strip(), 'time': row[1].strip(), 'date': d.isoformat()})
    return out


def mark_attendance(name):
    """Returns 'recorded' or 'already'. One record per person per day."""
    now = datetime.now()
    with lock:
        if any(r['name'].upper() == name.upper() and r['date'] == now.date().isoformat() for r in read_records()):
            return 'already'
        with open(CSV_PATH, 'a') as f:
            f.write(f"\n{name},{now:%H:%M:%S},{now:%d/%m/%Y}")
    return 'recorded'


class Scanner:
    def __init__(self):
        self.status, self.message, self.person = 'initializing', 'Loading enrolled faces.', None
        self.frame, self.events, self.running, self.skipped = None, [], False, []
        self.known, self.names, self.hold_until, self.cap = [], [], 0, None
        threading.Thread(target=self._load, daemon=True).start()

    def _set(self, status, message, person=None):
        self.status, self.message, self.person = status, message, person

    def _load(self):
        try:
            self._load_inner()
        except BaseException as e:  # includes SystemExit raised by face_recognition on import
            log.exception('Loading enrolled faces failed')
            self._set('error', f'Loading failed: {type(e).__name__}: {e}. See the Terminal running server.py for the full error.')

    def _load_inner(self):
        try:
            import cv2, numpy, face_recognition
        except (ImportError, SystemExit) as e:
            log.exception('Could not import face_recognition')
            self._set('error', f'face_recognition could not be imported ({type(e).__name__}: {e}). Check the Terminal running server.py; try: pip install "setuptools<81" face_recognition_models')
            return
        self.cv2, self.np, self.fr = cv2, numpy, face_recognition
        if not os.path.isdir(IMAGES):
            self._set('error', 'The Images_Attendance folder was not found next to server.py.')
            return
        for fn in sorted(os.listdir(IMAGES)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            img = cv2.imread(os.path.join(IMAGES, fn))
            enc = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else []
            if enc:
                self.known.append(enc[0]); self.names.append(os.path.splitext(fn)[0])
            else:
                self.skipped.append(fn)  # original code would crash here
        if not self.known:
            self._set('error', 'No usable faces found in Images_Attendance.')
        else:
            self._set('ready', 'Camera is off. Start the scanner to begin.')

    def start(self):
        if self.running or not self.known:
            return False
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def stop(self):
        self.running = False

    def _run(self):
        try:
            self._run_inner()
        except Exception as e:
            log.exception('Scanner loop crashed')
            self._set('error', f'Scanner stopped: {type(e).__name__}: {e}')
        finally:
            if self.cap is not None:
                self.cap.release(); self.cap = None
            self.running, self.frame = False, None

    def _run_inner(self):
        cv2, np, fr = self.cv2, self.np, self.fr
        cap = self.cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.running = False
            self._set('error', 'Could not open the camera. Close other apps that use it, then try again.')
            return
        self._set('scanning', 'Looking for a face.')
        while self.running:
            ok, img = cap.read()
            if not ok:
                self._set('error', 'The camera stopped sending frames.'); break
            small = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)
            locs = fr.face_locations(small)
            encs = fr.face_encodings(small, locs)
            holding = time.time() < self.hold_until
            if not locs and not holding:
                self._set('scanning', 'Looking for a face.')
            for enc, (y1, x2, y2, x1) in zip(encs, locs):
                dis = fr.face_distance(self.known, enc)
                i = int(np.argmin(dis))
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                if fr.compare_faces(self.known, enc)[i]:
                    name = self.names[i].upper()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    if not holding:
                        result = mark_attendance(name)
                        self.hold_until = time.time() + 3
                        self.events = ([{'name': name, 'result': result, 'at': datetime.now().strftime('%H:%M:%S')}] + self.events)[:10]
                        if result == 'recorded':
                            self._set('recorded', f'Attendance recorded for {name}.', name)
                        else:
                            self._set('already_recorded', f'{name} was already recorded today.', name)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if not holding:
                        self._set('failed', 'A face was found but did not match anyone enrolled.')
            ok, buf = cv2.imencode('.jpg', img)
            if ok:
                self.frame = buf.tobytes()
        cap.release()
        self.running, self.frame = False, None
        if self.status != 'error':
            self._set('ready', 'Camera is off. Start the scanner to begin.')


scanner = Scanner()


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    log.exception('Unhandled error on %s %s', request.method, request.path)
    return jsonify(error=f'{type(e).__name__}: {e}'), 500


def parse_date(raw):
    try:
        return (datetime.fromisoformat(raw).date() if raw else datetime.now().date()).isoformat()
    except ValueError:
        return None


@app.get('/api/health')
def health():
    ros = roster()
    return jsonify(ok=True, roster_available=ros is not None, csv_available=os.path.exists(CSV_PATH),
                   scanner=scanner.status, skipped_images=scanner.skipped)


@app.get('/api/students')
def students():
    ros = roster()
    if ros is None:
        return jsonify(error='Images_Attendance folder not found'), 404
    days, last = defaultdict(set), {}
    for r in read_records():
        k = r['name'].upper(); days[k].add(r['date']); last[k] = max(last.get(k, ''), r['date'])
    return jsonify(students=[{'name': n, 'days_present': len(days[n.upper()]), 'last_present': last.get(n.upper())} for n in ros])


@app.get('/api/attendance')
def attendance():
    date = request.args.get('date')
    if date and not parse_date(date):
        return jsonify(error='date must be YYYY-MM-DD'), 400
    recs = [r for r in read_records() if not date or r['date'] == date]
    return jsonify(records=sorted(recs, key=lambda r: (r['date'], r['time']), reverse=True))


@app.get('/api/summary')
def summary():
    date = parse_date(request.args.get('date'))
    if not date:
        return jsonify(error='date must be YYYY-MM-DD'), 400
    recs, ros = read_records(), roster()
    by_date = defaultdict(set)
    for r in recs:
        by_date[r['date']].add(r['name'].upper())
    ros_up = {n.upper() for n in ros} if ros is not None else None
    present = by_date.get(date, set())
    present_n = len(present & ros_up) if ros_up is not None else len(present)
    total = len(ros_up) if ros_up is not None else None
    return jsonify(date=date, roster_available=ros is not None, total_students=total, present=present_n,
                   absent=(total - present_n) if total is not None else None,
                   percentage=round(present_n / total * 100, 1) if total else None,
                   per_date=[{'date': d, 'present': len(n)} for d, n in sorted(by_date.items())])


@app.get('/api/scanner/status')
def scanner_status():
    return jsonify(status=scanner.status, message=scanner.message, person=scanner.person,
                   running=scanner.running, events=scanner.events)


@app.post('/api/scanner/start')
def scanner_start():
    return jsonify(started=scanner.start(), status=scanner.status, message=scanner.message)


@app.post('/api/scanner/stop')
def scanner_stop():
    scanner.stop()
    return jsonify(stopped=True)


@app.get('/api/stream')
def stream():
    def gen():
        while scanner.running:
            if scanner.frame:
                yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + scanner.frame + b'\r\n'
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=f')


def camera_preflight():
    """macOS only asks for camera permission reliably from the main thread. Doing the first open here
    (instead of in the scanner thread) lets the permission prompt appear without killing the server."""
    if sys.platform != 'darwin' or os.environ.get('FACESCAN_SKIP_CAMERA_CHECK'):
        return
    try:
        import cv2
        cap = cv2.VideoCapture(0); ok = cap.isOpened(); cap.release()
        log.info('Camera check: %s', 'camera opened' if ok else 'camera could NOT be opened (allow camera access for Terminal in System Settings > Privacy & Security > Camera, then restart)')
    except Exception:
        log.exception('Camera check failed')


if __name__ == '__main__':
    ros = roster()
    log.info('Project folder : %s', BASE)
    log.info('Images folder  : %s (%s)', IMAGES, 'missing' if ros is None else f'{len(ros)} images')
    log.info('Attendance CSV : %s (%s)', CSV_PATH, 'found' if os.path.exists(CSV_PATH) else 'missing')
    camera_preflight()
    log.info('Serving on http://127.0.0.1:%s', PORT)
    app.run(host='127.0.0.1', port=PORT, threaded=True)

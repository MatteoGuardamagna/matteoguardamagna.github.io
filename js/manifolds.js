/*
 * Neural manifold animation.
 *
 * A cloud of points (possible population states) with a bump of activity
 * travelling across it. The cloud starts structureless, condenses onto an
 * S-shaped line, curls shut into a ring and inflates into a torus.
 * All shapes are a tube around one curve whose curvature is blended from an
 * S (two opposite bends) to a constant (a closed ring); a fat tube on the ring
 * is the torus.
 *
 * No dependencies. Respects prefers-reduced-motion and only draws while visible.
 */
(function () {
    'use strict';

    var TAU = Math.PI * 2;
    var THIN = 0.07;   // tube radius of the line and the ring
    var FAT = 0.46;    // tube radius of the torus (ring radius is 1)
    var S_BEND = 1.2;  // curvature of each lobe of the S (each lobe turns ~200°)
    var SIGMA_U = 0.04, SIGMA_V = 0.09;   // width of the activity bump
    var BUCKETS = 6;   // depth shading levels
    var TRAIL = 28;
    var N = 241;       // samples along the curve

    var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    var darkScheme = window.matchMedia('(prefers-color-scheme: dark)');

    // ---------- helpers ----------
    function rng(seed) {
        return function () {
            seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    function gauss(rand) {
        return Math.sqrt(-2 * Math.log(rand() || 1e-9)) * Math.cos(TAU * rand());
    }
    function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
    function lerp(a, b, t) { return a + (b - a) * t; }
    function frac(x) { return x - Math.floor(x); }
    function wrap(d) { return d - Math.round(d); }
    function ease(x) {
        x = clamp01(x);
        return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    }
    function smooth(e0, e1, x) {
        var t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3 - 2 * t);
    }
    function rgba(hex, a) {
        var h = hex.replace('#', '');
        if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        var n = parseInt(h, 16);
        return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')';
    }

    // ---------- geometry ----------
    // The centre curve, sampled at N points: position (CX, CZ) and heading CT.
    var CX = new Float32Array(N), CZ = new Float32Array(N), CT = new Float32Array(N);

    // Curvature at arc length s in [-π, π]: an S blended toward a unit ring.
    function curvature(s, bend) {
        return (1 - bend) * S_BEND * Math.tanh(s / 0.35) + bend;
    }

    // Integrates the curve for a given bend and returns its radius around the centre.
    function buildCurve(bend) {
        var ds = TAU / (N - 1), mid = (N - 1) / 2, i;
        CT[0] = 0;
        for (i = 1; i < N; i++) {
            var s = -Math.PI + i * ds;
            CT[i] = CT[i - 1] + 0.5 * (curvature(s - ds, bend) + curvature(s, bend)) * ds;
        }
        var shift = -CT[mid];   // the middle stroke runs left to right, so the S stands upright
        for (i = 0; i < N; i++) CT[i] += shift;
        CX[0] = 0; CZ[0] = 0;
        for (i = 1; i < N; i++) {
            CX[i] = CX[i - 1] + 0.5 * (Math.cos(CT[i - 1]) + Math.cos(CT[i])) * ds;
            CZ[i] = CZ[i - 1] + 0.5 * (Math.sin(CT[i - 1]) + Math.sin(CT[i])) * ds;
        }
        var mx = 0, mz = 0;
        for (i = 0; i < N - 1; i++) { mx += CX[i]; mz += CZ[i]; }
        mx /= N - 1; mz /= N - 1;
        var r = 0;
        for (i = 0; i < N; i++) {
            CX[i] -= mx; CZ[i] -= mz;
            r = Math.max(r, Math.sqrt(CX[i] * CX[i] + CZ[i] * CZ[i]));
        }
        return r;
    }

    // Point (u, v) on a tube of radius `tube` around the current curve.
    function shape(u, v, tube, out) {
        var f = clamp01(u) * (N - 1), i = Math.min(N - 2, f | 0), a = f - i;
        var x = CX[i] + (CX[i + 1] - CX[i]) * a;
        var z = CZ[i] + (CZ[i + 1] - CZ[i]) * a;
        var th = CT[i] + (CT[i + 1] - CT[i]) * a;
        var c = Math.cos(TAU * v), sn = Math.sin(TAU * v);
        out[0] = x - tube * c * Math.sin(th);
        out[1] = tube * sn;
        out[2] = z + tube * c * Math.cos(th);
    }

    // Where the state wanders while there is no structure yet.
    function noiseWalk(t, out) {
        out[0] = 0.75 * Math.sin(0.53 * t + 1);
        out[1] = 0.55 * Math.sin(0.71 * t + 2);
        out[2] = 0.75 * Math.sin(0.37 * t + 0.4);
    }

    // ---------- timeline ----------
    // structureless cloud -> S-shaped line -> ring -> torus -> cloud.
    // [time, noise, bend, tube, pitch]; pitch looks down more on the S so it reads.
    var KEYS = [
        [0, 1, 0, THIN, 0.62],
        [3.5, 1, 0, THIN, 1.2],
        [6.5, 0, 0, THIN, 1.2],
        [9.5, 0, 0, THIN, 1.2],
        [12.5, 0, 1, THIN, 0.62],
        [15, 0, 1, THIN, 0.62],
        [18, 0, 1, FAT, 0.62],
        [23.5, 0, 1, FAT, 0.62],
        [26.5, 1, 1, FAT, 0.62]
    ];
    var PERIOD = KEYS[KEYS.length - 1][0];

    function params(t) {
        t = frac(t / PERIOD) * PERIOD;
        for (var i = 1; i < KEYS.length; i++) {
            if (t <= KEYS[i][0]) {
                var a = KEYS[i - 1], b = KEYS[i];
                var e = ease((t - a[0]) / (b[0] - a[0]));
                return {
                    noise: lerp(a[1], b[1], e), bend: lerp(a[2], b[2], e),
                    tube: lerp(a[3], b[3], e), pitch: lerp(a[4], b[4], e)
                };
            }
        }
        return { noise: 1, bend: 0, tube: THIN, pitch: 0.62 };
    }

    // Position of the activity bump on the manifold at time t.
    function state(t) { return [frac(0.045 * t), frac(0.11 * t)]; }

    var COUNT = 2400, STILL = 20, START = 1.5;

    // ---------- view ----------
    function View(canvas, stageEls) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.stageEls = stageEls;
        this.stage = -1;
        this.w = 0; this.h = 0;
        this.visible = false;
        this.dirty = true;
        this.yawOff = 0; this.pitchOff = 0;

        var n = COUNT, rand = rng(11);
        this.n = n;
        this.u = new Float32Array(n);
        this.v = new Float32Array(n);
        this.g = new Float32Array(n * 3);     // position in the structureless cloud
        this.ph = new Float32Array(n * 4);    // flicker phase/rate, jitter phase/rate
        for (var i = 0; i < n; i++) {
            this.u[i] = rand();
            this.v[i] = rand();
            this.g[3 * i] = 0.62 * gauss(rand);
            this.g[3 * i + 1] = 0.62 * gauss(rand);
            this.g[3 * i + 2] = 0.62 * gauss(rand);
            this.ph[4 * i] = rand() * TAU;
            this.ph[4 * i + 1] = 0.5 + 1.8 * rand();
            this.ph[4 * i + 2] = rand() * TAU;
            this.ph[4 * i + 3] = 0.4 + 1.2 * rand();
        }
        this.sx = new Float32Array(n);
        this.sy = new Float32Array(n);
        this.sr = new Float32Array(n);
        this.sb = new Uint8Array(n);
        this.act = new Float32Array(n);
        this.tmp = [0, 0, 0];
        this.walk = [0, 0, 0];
        this.trail = new Float32Array((TRAIL + 1) * 2);

        this.bindPointer();
    }

    View.prototype.resize = function () {
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.w = w; this.h = h;
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.dirty = true;
    };

    View.prototype.bindPointer = function () {
        var self = this, drag = null;
        this.canvas.addEventListener('pointerdown', function (e) {
            drag = { x: e.clientX, y: e.clientY };
            try { self.canvas.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        });
        this.canvas.addEventListener('pointermove', function (e) {
            if (!drag) return;
            self.yawOff += (e.clientX - drag.x) * 0.008;
            self.pitchOff = Math.max(-0.6, Math.min(0.8, self.pitchOff + (e.clientY - drag.y) * 0.006));
            drag.x = e.clientX; drag.y = e.clientY;
            self.dirty = true;
        });
        function end() { drag = null; }
        this.canvas.addEventListener('pointerup', end);
        this.canvas.addEventListener('pointercancel', end);
    };

    View.prototype.draw = function (t, colors) {
        var w = this.w, h = this.h;
        if (!w || !h) return;
        var ctx = this.ctx, n = this.n;
        var u = this.u, v = this.v, g = this.g, ph = this.ph;
        var sx = this.sx, sy = this.sy, sr = this.sr, sb = this.sb, act = this.act;
        var tmp = this.tmp, walk = this.walk;

        var p = params(t);
        var noise = p.noise, bend = p.bend, tube = p.tube;
        var ext = lerp(buildCurve(bend) + tube, 1.5, noise);
        var scale = Math.min(0.43 * w, 0.5 * h) / ext;
        var cx = w / 2, cy = h / 2;
        var yaw = 0.42 * Math.sin(0.16 * t) + this.yawOff;
        var pitch = p.pitch + 0.12 * Math.sin(0.11 * t + 1) + this.pitchOff;
        var cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
        var PERSP = 3.2;
        var base = Math.max(0.8, Math.min(1.25, w / 420));

        function project(x, y, z, i) {
            var x1 = x * cyw + z * syw;
            var z1 = -x * syw + z * cyw;
            var y2 = y * cp - z1 * sp;
            var z2 = y * sp + z1 * cp;
            var zn = z2 / ext;
            var f = PERSP / (PERSP - zn);
            sx[i] = cx + x1 * scale * f;
            sy[i] = cy - y2 * scale * f;
            return clamp01((zn + 1) / 2);
        }

        // Where the activity is now. On an open line it fades out at the ends.
        var st = state(t), su = st[0], sv = st[1];
        var closed = bend > 0.9;
        var wv = clamp01((tube - THIN) / (FAT - THIN));
        var fade = 1 - (1 - bend) * (1 - smooth(0.02, 0.14, Math.min(su, 1 - su)));
        var bumpGain = (1 - noise) * fade;
        var flickGain = Math.max(noise, 0.06);
        var jit = 0.012 + 0.07 * noise;
        var iu = 1 / (2 * SIGMA_U * SIGMA_U), iv = 1 / (2 * SIGMA_V * SIGMA_V);

        for (var i = 0; i < n; i++) {
            shape(u[i], v[i], tube, tmp);
            var jp = ph[4 * i + 2], jf = ph[4 * i + 3] * t;
            var x = lerp(tmp[0], g[3 * i], noise) + jit * Math.sin(jf + jp);
            var y = lerp(tmp[1], g[3 * i + 1], noise) + jit * Math.sin(1.3 * jf + 2 * jp);
            var z = lerp(tmp[2], g[3 * i + 2], noise) + jit * Math.sin(0.7 * jf + 3 * jp);
            var d = project(x, y, z, i);
            sr[i] = (0.8 + 0.9 * d) * base;
            sb[i] = Math.min(BUCKETS - 1, (d * BUCKETS) | 0);

            var du = u[i] - su;
            if (closed) du = wrap(du);
            var dv = wrap(v[i] - sv) * wv;
            var bump = Math.exp(-du * du * iu - dv * dv * iv) * bumpGain;
            var s = Math.sin(ph[4 * i] + ph[4 * i + 1] * t);
            var flick = s > 0.99 ? Math.pow(s, 400) * flickGain : 0;   // brief, sparse spikes
            act[i] = bump > flick ? bump : flick;
        }

        ctx.clearRect(0, 0, w, h);

        // Base points, batched by depth.
        ctx.fillStyle = colors.point;
        for (var b = 0; b < BUCKETS; b++) {
            ctx.globalAlpha = lerp(0.14, 0.62, (b + 0.5) / BUCKETS);
            ctx.beginPath();
            for (i = 0; i < n; i++) {
                if (sb[i] !== b) continue;
                ctx.moveTo(sx[i] + sr[i], sy[i]);
                ctx.arc(sx[i], sy[i], sr[i], 0, TAU);
            }
            ctx.fill();
        }

        // Active points.
        ctx.fillStyle = colors.accent;
        for (i = 0; i < n; i++) {
            var a = act[i];
            if (a < 0.04) continue;
            ctx.globalAlpha = Math.min(1, a * 1.1);
            ctx.beginPath();
            ctx.arc(sx[i], sy[i], sr[i] * (1 + 0.9 * a), 0, TAU);
            ctx.fill();
        }

        // The current state and its recent path.
        var dotA = (1 - noise) * fade + noise * 0.5;
        if (dotA > 0.02) {
            var tr = this.trail, prevU = su, broken = TRAIL + 1;
            for (var k = 0; k <= TRAIL; k++) {
                var tk = t - k * 0.045;
                var sk = state(tk);
                if (!closed && Math.abs(sk[0] - prevU) > 0.5) { broken = k; break; }
                prevU = sk[0];
                shape(sk[0], sk[1], tube * wv, tmp);
                noiseWalk(tk, walk);
                project(lerp(tmp[0], walk[0], noise), lerp(tmp[1], walk[1], noise), lerp(tmp[2], walk[2], noise), 0);
                tr[2 * k] = sx[0];
                tr[2 * k + 1] = sy[0];
            }
            ctx.strokeStyle = colors.accent;
            ctx.lineCap = 'round';
            for (k = broken - 1; k >= 1; k--) {
                var q = k / TRAIL;
                ctx.globalAlpha = dotA * 0.85 * Math.pow(1 - q, 1.5);
                ctx.lineWidth = lerp(3, 0.6, q) * base;
                ctx.beginPath();
                ctx.moveTo(tr[2 * k], tr[2 * k + 1]);
                ctx.lineTo(tr[2 * k - 2], tr[2 * k - 1]);
                ctx.stroke();
            }
            var hx = tr[0], hy = tr[1], hr = 18 * base;
            var halo = ctx.createRadialGradient(hx, hy, 0, hx, hy, hr);
            halo.addColorStop(0, rgba(colors.accent, 0.38));
            halo.addColorStop(1, rgba(colors.accent, 0));
            ctx.globalAlpha = dotA;
            ctx.fillStyle = halo;
            ctx.beginPath();
            ctx.arc(hx, hy, hr, 0, TAU);
            ctx.fill();
            ctx.fillStyle = colors.accent;
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(hx, hy, 4.2 * base, 0, TAU);
            ctx.fill();
            ctx.stroke();
        }
        ctx.globalAlpha = 1;

        if (this.stageEls) {
            var stage = noise > 0.5 ? 0 : bend < 0.5 ? 1 : tube < (THIN + FAT) / 2 ? 2 : 3;
            if (stage !== this.stage) {
                this.stage = stage;
                for (var j = 0; j < this.stageEls.length; j++) {
                    this.stageEls[j].classList.toggle('active', j === stage);
                }
            }
        }
    };

    // ---------- wiring ----------
    function init() {
        var canvas = document.getElementById('hero-manifold');
        if (!canvas || !window.requestAnimationFrame || !canvas.getContext) return;
        var view = new View(canvas, document.querySelectorAll('.stages li'));

        var colors = {};
        function readColors() {
            var cs = getComputedStyle(document.documentElement);
            colors.point = cs.getPropertyValue('--point').trim() || '#15203a';
            colors.accent = cs.getPropertyValue('--accent').trim() || '#d9572b';
            colors.bg = cs.getPropertyValue('--bg').trim() || '#f7f5f0';
            view.dirty = true;
        }
        readColors();
        if (darkScheme.addEventListener) darkScheme.addEventListener('change', readColors);

        if ('ResizeObserver' in window) new ResizeObserver(function () { view.resize(); }).observe(canvas);
        else window.addEventListener('resize', function () { view.resize(); });
        view.resize();

        if ('IntersectionObserver' in window) {
            new IntersectionObserver(function (entries) {
                view.visible = entries[entries.length - 1].isIntersecting;
                view.dirty = true;
            }, { rootMargin: '120px' }).observe(canvas);
        } else {
            view.visible = true;
        }

        // The clock only runs while the canvas is on screen, so the story starts when it is seen.
        var clock = START, last = null;
        function frame(now) {
            var animate = !reduceMotion.matches;
            if (last !== null && view.visible) clock += Math.min(0.1, (now - last) / 1000);
            last = now;
            if (view.visible && (animate || view.dirty)) {
                view.draw(animate ? clock : STILL, colors);
                view.dirty = false;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();

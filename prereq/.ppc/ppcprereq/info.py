import math

code = "prereq"
name = "Pre"
descr = "prerequisite test"


def readable(x):
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    i = 0
    while x != 0 and float(f'{x:.{i}f}') == 0:
        i += 1
    i += 2
    return f'{x:.{i}f}'


def isnum(v):
    return v is not None and (isinstance(v, int) or
                              (isinstance(v, float) and math.isfinite(v)))


def safeget3(m, i, j, k):
    try:
        return m[i][j][k]
    except IndexError:
        return None
    except TypeError:
        return None


def safenum(v, default=0):
    if isnum(v):
        return v
    else:
        return default


def saferatio(x, y, f):
    if isnum(x) and isnum(y):
        return x < f * y
    else:
        return False


def safeprint(v, fmt='{:+.8f}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return str(v)


def safeprintt(v, fmt='{:+.8f}', sfmt='{:>11s}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return sfmt.format(v)


def safereadable(v):
    if v is None:
        return '–'
    elif isnum(v):
        return readable(v)
    else:
        return str(v)


def html():
    from markupsafe import Markup
    return Markup("""
<p>You need to write a function that takes as input a bitmap image and the coordinates of a rectangle, and it has to calculate the <strong>average color</strong> of all pixels inside the rectangle.</p>

<h3>Interface</h3>

<p>We have already defined the following type for storing the result:</p>
<div class="prewrap"><pre>
struct Result {
    float avg[3];
};
</pre></div>
<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
Result calculate(int ny, int nx, const float *data,
                 int y0, int x0, int y1, int x1)
</pre></div>
<p>Here <code>data</code> is a color image with <code>ny*nx</code> pixels, and each pixel consists of three color components, red, green, and blue. In total, there are <code>ny*nx*3</code> floating point numbers in the array <code>data</code>.</p>

<p>The color components are numbered <code>0 &lt;= c &lt; 3</code>, x coordinates are numbered <code>0 &lt;= x &lt; nx</code>, y coordinates are numbered <code>0 &lt;= y &lt; ny</code>, and the value of this color component is stored in <code>data[c&nbsp;+&nbsp;3&nbsp;*&nbsp;x&nbsp;+&nbsp;3&nbsp;*&nbsp;nx&nbsp;*&nbsp;y]</code>.</p>

<p>The parameters <code>y0</code>, <code>x0</code>, <code>y1</code>, and <code>x1</code> indicate the <strong>location</strong> of the rectangle. The upper left corner of the rectangle is at coordinates (<code>x0</code>,&nbsp;<code>y0</code>), and the lower right corner is at coordinates (<code>x1-1</code>,&nbsp;<code>y1-1</code>). That is, the width of the rectangle is <code>x1-x0</code> pixels and the height is <code>y1-y0</code> pixels. The coordinates satisfy <code>0&nbsp;&lt;=&nbsp;y0&nbsp;&lt;&nbsp;y1&nbsp;&lt;=&nbsp;ny</code> and <code>0&nbsp;&lt;=&nbsp;x0&nbsp;&lt;&nbsp;x1&nbsp;&lt;=&nbsp;nx</code>.</p>

<p>In the result that you return, <code>avg[c]</code> has to contain the arithmetic mean of the color component <code>c</code> for all pixels inside the rectangle.</p>

<h3>Details</h3>

<p>Even though the input and output are single-precision floating-point numbers, you must do <strong>all arithmetic with double-precision floating point numbers</strong>, and only round the final result back to single precision.</p>

<p>You can assume that there are at most 10 million pixels in the input image.</p>
""")


def explain_web(test):
    from jinja2 import Template
    from markupsafe import Markup

    templ_basic = Template("""
{% if input.ny and input.nx %}
    <p>In this test I called your function with the following parameters:</p>
    <ul class="compact">
        <li>ny = {{ input.ny }}</li>
        <li>nx = {{ input.nx }}</li>
        <li>y0 = {{ input.y0 }}</li>
        <li>x0 = {{ input.x0 }}</li>
        <li>y1 = {{ input.y1 }}</li>
        <li>x1 = {{ input.x1 }}</li>
    </ul>
    {% if input.data %}
        <p>This is what the input data looked like; here we show the three color components of each pixel:</p>

        <div class="matrixwrap"><div class="matrix"><table>
            <tr>
                <td></td>{% for j in range(input.nx) %}<td class="colindex">{{ j }}</td>{% endfor %}
            </tr>
            {% for i in range(input.ny) %}
                <tr>
                    <td class="rowindex">{{ i }}</td>
                    {% for j in range(input.nx) %}
                        <td class="element">
                            {{ safeprint(safeget3(input.data,i,j,0)) }}<br>
                            {{ safeprint(safeget3(input.data,i,j,1)) }}<br>
                            {{ safeprint(safeget3(input.data,i,j,2)) }}
                        </td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </table></div></div>
    {% endif %}
{% endif %}

{% if output.avg %}
    <p>Your code produced the following result:</p>
    <ul class="compact">
    <li>avg[0] = {{ safeprint(output.avg[0]) }}</li>
    <li>avg[1] = {{ safeprint(output.avg[1]) }}</li>
    <li>avg[2] = {{ safeprint(output.avg[2]) }}</li>
    </ul>
{% endif %}

{% if safenum(oe.error_magnitude) > 0 and safenum(oe.threshold) > 0 %}
    <p>In comparison with the expected output, the largest error was ≈ <strong>{{ safereadable(oe.error_magnitude) }}</strong>.
    In this test the largest errors should be &lt {{ safereadable(oe.threshold) }}.
    That is, your code made errors that are ≈ {{ safereadable(oe.error_magnitude/oe.threshold) }} times too large.</p>
    {% if saferatio(oe.error_magnitude, oe.threshold, 100) %}
        <p>As the errors were relatively small, could they be maybe rounding errors? Perhaps you could double-check that you have done all arithmetic in double precision?</p>
    {% endif %}
{% endif %}
""")
    return Markup(
        templ_basic.render(
            input=test.raw.get("input", {}),
            output=test.raw.get("output", {}),
            oe=test.raw.get("output_errors", {}),
            safeget3=safeget3,
            safenum=safenum,
            saferatio=saferatio,
            safeprint=safeprint,
            safereadable=safereadable,
        ))


def explain_terminal(r, color=False):
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}
    error_magnitude = oe.get('error_magnitude', None)
    threshold = oe.get('threshold', None)

    if color:
        hl, reset = '\033[31;1m', '\033[0m'
    else:
        hl, reset = '', ''

    expl = ''
    par = ['ny', 'nx', 'y0', 'x0', 'y1', 'x1']
    if all(x in input for x in par):
        expl += f'In this test I called your function with the following parameters:\n'
        for x in par:
            expl += f' · {x} = {input[x]}\n'
        ny = input['ny']
        nx = input['nx']
        expl += '\n'
        if 'data' in input:
            expl += f'This is what the input data looked like:\n'
            for y in range(ny):
                for x in range(nx):
                    v = input['data'][y][x]
                    expl += f' · pixel at y ={y:3d}, x ={x:3d}:  ({v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f})\n'
            expl += '\n'
    if 'avg' in output:
        v = output['avg']
        if r.errors:
            expl += f'Your code produced the following result, which was {hl}not correct{reset}:\n'
        else:
            expl += f'Your code produced the following result:\n'
        expl += f' · avg:  ({v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f})\n'
        expl += '\n'

    if safenum(error_magnitude) > 0 and safenum(threshold) > 0:
        expl += f'In comparison with the expected output, the largest error was ≈ {hl}{readable(error_magnitude)}{reset}.\n'
        expl += f'In this test the largest errors should be < {readable(threshold)}.\n'
        rel = error_magnitude / threshold
        expl += f'That is, your code made errors that are ≈ {readable(rel)} times too large.\n'
        if rel < 100:
            expl += f'As the errors were relatively small, could they be maybe rounding errors?\n'
            expl += f'Perhaps you could double-check that you have done all arithmetic in\n'
            expl += f'double precision?\n'
    expl += '\n'

    return expl

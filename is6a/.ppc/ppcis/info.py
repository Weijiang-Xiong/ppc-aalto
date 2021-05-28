import math

code = "is"
name = "IS"
descr = "image segmentation"


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


def safeget(m, i):
    try:
        return m[i]
    except KeyError:
        return None
    except IndexError:
        return None
    except TypeError:
        return None


def safeget2(m, i, j):
    try:
        return m[i][j]
    except KeyError:
        return None
    except IndexError:
        return None
    except TypeError:
        return None


def safeget3(m, i, j, k):
    try:
        return m[i][j][k]
    except KeyError:
        return None
    except IndexError:
        return None
    except TypeError:
        return None


def safenum(v, default=0):
    if isnum(v):
        return v
    else:
        return default


def safestr(v):
    if v is None:
        return '–'
    else:
        return str(v)


def safeprint(v, fmt='{:+.8f}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return str(v)


def safeprintt(v, fmt='{:+.8f}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return str(v)


def html():
    from markupsafe import Markup

    base = '/static/exercises/is/'
    examples = ''
    for f in [1, 2, 3, 4, 5]:
        examples += f'''<div class="figpair"><a href="{base}{f}a.png" class="plain"><img alt="Input" class="figpair figpairleft" src="{base}{f}a.png" title="Input"></a> <a href="{base}{f}b.png" class="plain"><span class="swap"><img alt="Output" class="figpair figpairright" src="{base}{f}b.png" title="Output"><img alt="Output" class="figpair figpairright" src="{base}{f}c.png" title="Output"></span></a></div>'''

    return Markup(r"""
<p>Find the best way to partition the given figure in two parts: a monochromatic rectangle and a monochromatic background. The objective is to minimize the sum of squared errors.</p>

<h3>Interface</h3>

<p>We have already defined the following type for storing the result:</p>
<div class="prewrap"><pre>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};
</pre></div>
<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
Result segment(int ny, int nx, const float* data)
</pre></div>
<p>Here <code>data</code> is a color image with <code>ny*nx</code> pixels, and each pixel consists of three color components, red, green, and blue. In total, there are <code>ny*nx*3</code> floating point numbers in the array <code>data</code>.</p>

<p>The color components are numbered <code>0 &lt;= c &lt; 3</code>, x coordinates are numbered <code>0 &lt;= x &lt; nx</code>, y coordinates are numbered <code>0 &lt;= y &lt; ny</code>, and the value of this color component is stored in <code>data[c + 3 * x + 3 * nx * y]</code>.</p>

<h3>Correct output</h3>

<p>In the <code>Results</code> structure, the first four fields indicate the <strong>location</strong> of the rectangle. The upper left corner of the rectangle is at coordinates (<code>x0</code>, <code>y0</code>), and the lower right corner is at coordinates (<code>x1-1</code>, <code>y1-1</code>). That is, the width of the rectangle is <code>x1-x0</code> pixels and the height is <code>y1-y0</code> pixels. The coordinates have to satisfy <code>0 &lt;= y0 &lt; y1 &lt;= ny</code> and <code>0 &lt;= x0 &lt; x1 &lt;= nx</code>.</p>

<p>The last two fields indicate the <strong>color</strong> of the background and the rectangle. Field <code>outer</code> contains the three color components of the background and field <code>inner</code> contains the three color components of the rectangle.</p>

<h3>Objective function</h3>

<p>For each pixel (<code>x,y</code>) and color component <code>c</code>, we define the error <code>error(y,x,c)</code> as follows:</p>
<ul class="compact">
<li>Let <code>color(y,x,c) = data[c + 3 * x + 3 * nx * y]</code>.</li>
<li>If (<code>x,y</code>) is located outside the rectangle: <code>error(y,x,c) = outer[c] - color(y,x,c)</code>.</li>
<li>If (<code>x,y</code>) is located inside the rectangle: <code>error(y,x,c) = inner[c] - color(y,x,c)</code>.</li>
</ul>
<p>The total <strong>cost</strong> of the segmentation is the <strong>sum of squared errors</strong>, that is, the sum of <code>error(y,x,c) * error(y,x,c)</code> over all <code>0 &lt;= c &lt; 3</code> and <code>0 &lt;= x &lt; nx</code> and <code>0 &lt;= y &lt; ny</code>.</p>

<p>Your task is to find a segmentation that minimizes the total cost.</p>

<h3>Algorithm</h3>

<p>In IS4, IS6a, and IS6b tasks you are expected to use an algorithm that tries out all possible locations 0 ≤ y0 < y1 ≤ ny and 0 ≤ x0 < x1 ≤ nx for the rectangle and finds the best one. However, for each candidate location you should only perform O(1) operations to evaluate how good this position is. To achieve this, some preprocessing will be needed.</p>

<p>In IS9a you are expected to design a more efficient algorithm that (at least in typical cases) does not need to try out all possible locations of the rectangle. In IS9a your submission will be graded using a structured input that might resemble e.g. a real-world image in which some candidate positions are much better than others.</p>

<h3>Examples</h3>

<p>These examples show the segmentation produced by a correct implementation (right) for some sample images (left). Hover the mouse on the output to better see the segmentation.</p>

""" + examples + """

<h3>General hints</h3>

<div class="spoiler">
<p>Spend some time with pen and paper first to get the math right. You need to find a very efficient way of checking which of the candidate positions are best. There is a fairly straightforward solution in which for each candidate position you will only need to do a few array lookups (to a precomputed array) and a few arithmetic operations to calculate an indicator for the quality of this position.</p>

<p>Remember that the average minimizes the sum of squared errors.</p>

<p>Use the inclusion–exclusion principle to quickly calculate the sum of values within a rectangle — if you can look up the orange and blue sums from a precomputed array, you can also calculate the sum of the gray area:</p>

<div class="figfull"><a href="/static/exercises/is/hint.png" class="plain"><img class="figfull" src="/static/exercises/is/hint.png" title="Applying the inclusion-exclusion principle in IS"></a></div>

<p>It may be helpful to organise the loops so that the outer loop tries out different <strong>sizes</strong> of the rectangle, and the inner loop tries all possible <strong>positions</strong> of the rectangle. Then in the innermost loop you only need to be able to compare candidate positions for rectangles with the same size. Precompute everything that you can outside the innermost loops.</p>
</div>

<h3>Hints for IS9a</h3>

<div class="spoiler">
<p>Here is one approach: First use a coarse grid, with e.g. 10 × 10 pixels per grid cell. Then try all possible ways to place the rectangle in this coarse grid. Each coarse location represents a set of approx. 10000 fine-grained locations (so it is much faster to try out all possible coarse locations). For each coarse location, calculate the following two estimates (efficiently, in constant time):</p>
<ul>
<li><strong>An upper bound:</strong> at most how much is the cost if I place the rectangle in some of the fine-grained locations that are represented by this coarse-grained location?</li>
<li><strong>A lower bound:</strong> at least how much is the cost if I place the rectangle in some of the fine-grained locations that are represented by this coarse-grained location?</li>
</ul>
<p>After these calculations, you can hopefully rule out a majority of coarse-grained locations: you know that there are other locations where the cost is <strong>at most</strong> some value <i>s</i>, so you can then ignore all coarse-grained locations where the cost is <strong>larger than</strong> <i>s</i>.</p>

<p>Finally, zoom in to those coarse-grained locations that you have not yet ruled out.</p>
</div>

""")


def explain_web(test):
    from jinja2 import Template
    from markupsafe import Markup

    templ_basic = Template("""
{% if input.nx and input.ny %}
    <p>In this test I called your function with ny = {{input.ny}} and nx = {{input.nx}}.</p>
    {% if input.data %}
        <p>This is what the input data looked like:</p>

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
{% if oe.wrong_output %}
    <p>Here is a comparison of your output and the expected output:</p>
    <div class="tablewrap"><table>
        <tr>
            <th></th>
            <th class="right">Output</th>
            <th class="right">Expected</th>
        </tr>
        {% for k in ['y0', 'x0', 'y1', 'x1'] %}
            {% if safeget(output, k) != safeget(expected, k) %}
                <tr>
                    <td>{{k}}</td>
                    <td class="right"><strong>{{safestr(safeget(output, k))}}</strong></td>
                    <td class="right"><strong>{{safestr(safeget(expected, k))}}</strong></td>
                </tr>
            {% else %}
                <tr>
                    <td>{{k}}</td>
                    <td class="right">{{safestr(safeget(output, k))}}</td>
                    <td class="right">{{safestr(safeget(expected, k))}}</td>
                </tr>
            {% endif %}
        {% endfor %}
        {% for k in ['inner', 'outer'] %}
            {% for i in [0, 1, 2] %}
                {% if too_large(output, expected, k, i) %}
                    <tr>
                        <td>{{k}}[{{i}}]</td>
                        <td class="right"><strong>{{safeprint(safeget2(output, k, i))}}</strong></td>
                        <td class="right"><strong>{{safeprint(safeget2(expected, k, i))}}</strong></td>
                    </tr>
                {% else %}
                    <tr>
                        <td>{{k}}[{{i}}]</td>
                        <td class="right">{{safeprint(safeget2(output, k, i))}}</td>
                        <td class="right">{{safeprint(safeget2(expected, k, i))}}</td>
                    </tr>
                {% endif %}
            {% endfor %}
        {% endfor %}
    </table></div>
{% endif %}
""")

    def too_large(output, expected, k, i):
        output = test.raw.get("output", {})
        oe = test.raw.get("output_errors", {})
        expected = test.raw.get("output_errors", {}).get("expected", {})
        threshold = safenum(oe.get('threshold'))
        a = safeget2(output, k, i)
        b = safeget2(expected, k, i)
        return abs(safenum(a) - safenum(b)) > threshold > 0

    return Markup(
        templ_basic.render(
            input=test.raw.get("input", {}),
            output=test.raw.get("output", {}),
            oe=test.raw.get("output_errors", {}),
            expected=test.raw.get("output_errors", {}).get("expected", {}),
            safeget=safeget,
            safeget2=safeget2,
            safeget3=safeget3,
            safenum=safenum,
            safeprint=safeprint,
            safestr=safestr,
            too_large=too_large,
        ))


def explain_terminal(r, color=False):
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}
    expected = oe.get('expected') or {}
    threshold = safenum(oe.get('threshold'))

    if color:
        hl, reset = '\033[31;1m', '\033[0m'
    else:
        hl, reset = '', ''

    expl = ''
    par = ['ny', 'nx']
    if all(x in input for x in par):
        ny = input['ny']
        nx = input['nx']
        expl += f'In this test I called your function with ny = {ny} and nx = {nx}.\n\n'
        if 'data' in input:
            data = input['data']
            expl += f'This is what the input data looked like:\n'
            for y in range(ny):
                for x in range(nx):
                    v = input['data'][y][x]
                    expl += f' · pixel at y ={y:3d}, x ={x:3d}:  ({safeprintt(safeget3(data,y,x,0))}, {safeprintt(safeget3(data,y,x,1))}, {safeprintt(safeget3(data,y,x,2))})\n'
            expl += '\n'
    if oe.get('wrong_output'):
        expl += f'Here is a comparison of your output and the expected output:\n\n'
        hfield, houtput, hcorrect = '', 'OUTPUT', 'EXPECTED'
        expl += f'{hfield:>15s} {houtput:>15s} {hcorrect:>15s}\n'
        for k in ['y0', 'x0', 'y1', 'x1']:
            a = safeget(output, k)
            b = safeget(expected, k)
            fmt = '{:d}'
            if a != b:
                expl += f'{k:>15s} {hl}{safeprintt(a, fmt):>15s} {safeprintt(b, fmt):>15s}{reset}\n'
            else:
                expl += f'{k:>15s} {safeprintt(a, fmt):>15s} {safeprintt(b, fmt):>15s}\n'
        for k in ['inner', 'outer']:
            for i in [0, 1, 2]:
                a = safeget2(output, k, i)
                b = safeget2(expected, k, i)
                label = f'{k}[{i}]'
                if abs(safenum(a) - safenum(b)) > threshold > 0:
                    expl += f'{label:>15s} {hl}{safeprintt(a):>15s} {safeprintt(b):>15s}{reset}\n'
                else:
                    expl += f'{label:>15s} {safeprintt(a):>15s} {safeprintt(b):>15s}\n'

    return expl

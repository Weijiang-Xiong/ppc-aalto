import math

code = "mf"
name = "MF"
descr = "median filter"


def html():
    from markupsafe import Markup

    base = '/static/exercises/mf/'
    examples = ''
    for k in [1, 2, 5, 10]:
        examples += f'''<p><em>k = {k}</em></p>'''
        for f in [1, 6]:
            examples += f'''<div class="figpair"><a href="{base}{f}a.png" class="plain"><img alt="Input" class="figpair figpairleft" src="{base}{f}a.png" title="Input"></a> <a href="{base}{f}b-{k}.png" class="plain"><span class="swap"><img alt="Output" class="figpair figpairright" src="{base}{f}b-{k}.png" title="Output"><img alt="Output" class="figpair figpairright" src="{base}{f}c-{k}.png" title="Output"></span></a></div>'''
    for k, n in [(1, 'x'), (2, 'y')]:
        examples += f'''<p><em>Noise reduction, k = {k}</em></p>'''
        for f in [1, 6]:
            examples += f'''<div class="figpair"><a href="{base}{f}{n}a.png" class="plain"><img alt="Input" class="figpair figpairleft" src="{base}{f}{n}a.png" title="Input"></a> <a href="{base}{f}{n}b-{k}.png" class="plain"><span class="swap"><img alt="Output" class="figpair figpairright" src="{base}{f}{n}b-{k}.png" title="Output"><img alt="Output" class="figpair figpairright" src="{base}{f}{n}c-{k}.png" title="Output"></span></a></div>'''

    return Markup(f"""
<p>In this task, we will implement a program for doing 2-dimensional median filtering with a rectangular window.</p>

<h3>Interface</h3>

<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
</pre></div>
<p>Here <code>in</code> and <code>out</code> are pointers to <code>float</code> arrays, each of them contains <code>ny*nx</code> elements. The arrays are already allocated by whoever calls this function; you do not need to do any memory management.</p>

<p>The original value of pixel (x,y) is given in <code>in[x + nx*y]</code> and its new value will be stored in <code>out[x + nx*y]</code>.</p>

<h3>Correct output</h3>
<p>In the output, pixel (x,y) will contain the median of all pixels with coordinates (i,j) where</p>
<ul class="compact">
<li><code>0 &lt;= i &lt; nx</code>,</li>
<li><code>0 &lt;= j &lt; ny</code>,</li>
<li><code>x - hx &lt;= i &lt;= x + hx</code>,</li>
<li><code>y - hy &lt;= j &lt;= y + hy</code>.</li>
</ul>
<p>This is the <em>sliding window</em>. Note that the window will contain at most <code>(2*hx+1) * (2*hy+1)</code> pixels, but near the boundaries there will be fewer pixels.</p>

<h3>Details</h3>

<p>For our purposes, the median of the vector <b>a</b> = (a<sub>1</sub>, a<sub>2</sub>, …, a<sub>n</sub>) is defined as follows:</p>
<ul class="compact">
<li>Let x<sub>1</sub>, x<sub>2</sub>, …, x<sub>n</sub> be the values of <b>a</b> sorted in a non-decreasing order.</li>
<li>If n = 2k + 1, then the median of <b>a</b> is x<sub>k+1</sub>.</li>
<li>If n = 2k, then the median of <b>a</b> is (x<sub>k</sub> + x<sub>k+1</sub>)/2.</li>
</ul>

<h3>Examples</h3>

<p>These examples show what a correct implementation will do if you apply it to each color component of a bitmap image. In these examples, for each color component, the value of each pixel is the median of all pixel values within a sliding window of dimensions (2k+1) × (2k+1). Hover the mouse on the output images to see the differences between input and output.</p>

{examples}

<h3>Hints for MF1–MF2</h3>

<div class="spoiler">
    <p>Make sure that you use a fast selection algorithm to find the median. Do not sort; <a href="http://en.wikipedia.org/wiki/Quickselect">quickselect</a> and other good selection algorithms are much faster than sorting, and there is already a good <a href="http://en.cppreference.com/w/cpp/algorithm/nth_element">implementation in the C++ standard library</a>, so you do not need to reinvent it.</p>
</div>

<h3>Hints for MF9a</h3>

<div class="spoiler">
    <p>Here are some examples of possible techniques that you can use to solve task <strong>MF9a</strong>:</p>
    <ul>
        <li>Partition the image in smaller, partially overlapping <strong>blocks</strong>. For example, if your windows size is 21 × 21, a reasonable block size might be 50 × 50. Solve the median filter problem separately for each block; place the blocks so that each output pixel comes from exactly one block. Be careful with the boundaries.</li>
        <li>Within each block, sort the input data and replace the original values by their ordinals, breaking ties arbitrarily — for example, if the input data was (1.0, 1.0, 0.1, 0.9, 0.5, 0.2, 1.0), replace it with (4, 5, 0, 3, 2, 1, 6). Now you have a much simpler task: instead of finding a median of floating point numbers, you only need to find the median of small, <strong>distinct</strong> integers.</li>
        <li>You can now represent the sliding window as a <strong>bit vector</strong>: for example, if your block size is 50 × 50, then your sliding window can be represented as a bit vector with 2500 bits. Do not recompute the bit vector from scratch for every window position — if you move the window from (x,y) to (x+1,y), you only need to reset 2k+1 bits and set 2k+1 bits.</li>
        <li>Exploit <strong>bit-parallelism</strong> and the special CPU instructions that help with bit manipulation. To find the median from the bit vector efficiently, you may want to have a look at e.g. compiler intrinsics <a href="https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html">__builtin_popcountll</a> (POPCNT instruction), <a href="https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html">__builtin_ctzll</a> (TZCNT instruction), and <a href="https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_pdep_u64&amp;expand=3894">_pdep_u64</a> (PDEP instruction). Some caching of pre-computed values may be helpful, too.</li>
        <li>Pay attention to the <strong>locality</strong> of memory references. For example, does it make more sense to slide the window from left to right or from top to bottom?</li>
        <li>You can use <strong>OpenMP</strong> to process multiple blocks in parallel.</li>
    </ul>
</div>
""")


def isnum(v):
    return v is not None and (isinstance(v, int) or
                              (isinstance(v, float) and math.isfinite(v)))


def safeget(m, i, j):
    try:
        return m[i][j]
    except IndexError:
        return None
    except TypeError:
        return None


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
        return sfmt.format(str(v))


def explain_web(test):
    from jinja2 import Template
    from markupsafe import Markup

    templ_basic = Template("""
{% if input.ny and input.nx and input.hy and input.hx %}
    <p>In this test I called your function with the following parameters:</p>
    <ul class="compact">
        <li>ny = {{ input.ny }}</li>
        <li>nx = {{ input.nx }}</li>
        <li>hy = {{ input.hy }}</li>
        <li>hx = {{ input.hx }}</li>
    </ul>
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
                        <td class="element">{{ safeprint(safeget(input.data,i,j)) }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </table></div></div>
    {% endif %}
    {% if output.result %}
        <p>This is the output that I got back:</p>
        <div class="matrixwrap"><div class="matrix"><table>
            <tr>
                <td></td>{% for j in range(input.ny) %}<td class="colindex">{{ j }}</td>{% endfor %}
            </tr>
            {% for i in range(input.ny) %}
                <tr>
                    <td class="rowindex">{{ i }}</td>
                    {% for j in range(input.nx) %}
                        {% if oe.locations and safeget(oe.locations,i,j) == 1 %}
                            <td class="element verywrong">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% else %}
                            <td class="element correct">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% endif %}
                    {% endfor %}
                </tr>
            {% endfor %}
        </table></div></div>
        {% if oe.locations %}
            <p>Above I have here <span class="elementexample verywrong">highlighted</span> the cells that contain wrong values.</p>
        {% endif %}
    {% endif %}
{% endif %}
""")
    return Markup(
        templ_basic.render(
            input=test.raw.get("input", {}),
            output=test.raw.get("output", {}),
            oe=test.raw.get("output_errors", {}),
            safeget=safeget,
            safeprint=safeprint,
        ))


def explain_terminal(r, color=False):
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    ny = input.get('ny', None)
    nx = input.get('nx', None)
    hy = input.get('hy', None)
    hx = input.get('hx', None)
    data = input.get('data', None)
    result = output.get("result", None)
    locations = oe.get("locations", None)

    if color:
        hl, reset = '\033[31;1m', '\033[0m'
        ok1, ok2 = '', ''
        error1, error2 = hl, reset
    else:
        hl, reset = '', ''
        ok1, ok2 = ' ', ' '
        error1, error2 = '[', ']'

    expl = ''
    if ny is not None and nx is not None and hy is not None and hx is not None:
        expl += f'In this test I called your function with the following parameters:\n'
        expl += f' · ny = {ny}\n'
        expl += f' · nx = {nx}\n'
        expl += f' · hy = {hy}\n'
        expl += f' · hx = {hx}\n'
        expl += '\n'
        if data is not None:
            expl += f'This is what the input data looked like:\n'
            for y in range(ny):
                expl += ' '
                for x in range(nx):
                    expl += ' ' + safeprintt(safeget(data, y, x))
                expl += '\n'
            expl += '\n'

        if result is not None:
            expl += f'This is the output that I got back:\n'
            for y in range(ny):
                expl += ' '
                for x in range(nx):
                    m1, m2 = '', ''
                    v = safeget(result, y, x)
                    if locations:
                        e = safeget(locations, y, x)
                        if e == 0:
                            m1, m2 = ok1, ok2
                        else:
                            m1, m2 = error1, error2
                    expl += f' {m1}{safeprintt(v)}{m2}'
                expl += '\n'
            expl += '\n'
            if locations is not None:
                expl += f'Above I have {error1}highlighted{error2} the cells that contain wrong values.\n'

    return expl

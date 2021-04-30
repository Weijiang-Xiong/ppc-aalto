import math

code = "cp"
name = "CP"
descr = "correlated pairs"


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


def safeget(m, i, j):
    try:
        return m[i][j]
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
        return sfmt.format(str(v))


def safereadable(v):
    if v is None:
        return '–'
    elif isnum(v):
        return readable(v)
    else:
        return str(v)


def html():
    from markupsafe import Markup

    base = '/static/exercises/cp/'
    examples = ''
    for f in [1, 2, 5, 7, 8]:
        examples += f'''<div class="figpair"><a href="{base}{f}d.png" class="plain"><img alt="Input" class="figpair figpairleft" src="{base}{f}d.png" title="Input"></a> <a href="{base}{f}e.png" class="plain"><img alt="Output" class="figpair figpairright" src="{base}{f}e.png" title="Output"></a></div>'''

    return Markup(f"""
<p>You are given m input vectors, each with n numbers. Your task is to calculate the <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient">correlation</a> between <em>every pair of input vectors</em>.</p>

<h3>Interface</h3>

<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
void correlate(int ny, int nx, const float* data, float* result)
</pre></div>
<p>Here <code>data</code> is a pointer to the input matrix, with <code>ny</code> rows and <code>nx</code> columns. For all <code>0 &lt;= y &lt; ny</code> and <code>0 &lt;= x &lt; nx</code>, the element at row <code>y</code> and column <code>x</code> is stored in <code>data[x + y*nx]</code>.</p>

<p>The function has to solve the following task: for all <code>i</code> and <code>j</code> with <code>0 &lt;= j &lt;= i &lt; ny</code>, calculate the <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient">correlation coefficient</a> between row <code>i</code> of the input matrix and row <code>j</code> of the input matrix, and store the result in <code>result[i + j*ny]</code>.</p>

<p>Note that the correlations are symmetric, so we will only compute the upper triangle of the result matrix. You can leave the lower triangle <code>i &lt; j</code> undefined.</p>

<p>The arrays <code>data</code> and <code>result</code> are already allocated by whoever calls this function; you do not need to do any memory management related to these arrays.</p>

<h3>Details</h3>

<p>The input and output are always given as single-precision floating point numbers (type <code>float</code>). However, depending on the task, we will do arithmetic with either single or double precision numbers:</p>
<ul>
    <li>If the task specifies that you must use double-precision floating point numbers, then <em>all</em> arithmetic operations must be done with type <code>double</code>, all intermediate results must be stored in variables of type <code>double</code>, and you will only round the final result to single precision.</li>
    <li>If the task specifies that you can use single-precision floating point numbers, then you are encouraged to use the type <code>float</code> whenever possible.</li>
</ul>
<p>However, in each case you will have to make sure the numerical precision of the results is sufficiently high. The grading tool makes sure the error is sufficiently small. The error thresholds are chosen so that a straightforward and efficient solution is typically good enough, but please feel free to ask the course staff for hints if you are struggling with the rounding errors.</p>

<h3>Examples</h3>

<p>These examples show what a correct implementation will do if you apply it to a bitmap image:</p>
<ul class="compact">
<li>Input (first image): vector i = row i of the image.</li>
<li>Output (second image): red pixel at (i, j) = positive correlation between rows i and j, blue pixel = negative correlation.</li>
</ul>

{examples}

<h3>Hints</h3>

<div class="spoiler">
    <p>A reasonable way to calculate all pairwise correlations is the following:</p>
    <ul class="compact">
        <li>First normalize the input rows so that each row has the arithmetic mean of 0 — be careful to do the normalization so that you do not change pairwise correlations.</li>
        <li>Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.</li>
        <li>Let X be the normalized input matrix.</li>
        <li>Calculate the (upper triangle of the) matrix product Y = XX<sup>T</sup>.</li>
    </ul>
    <p>Now matrix Y contains all pairwise correlations. The only computationally-intensive part is the computation of the matrix product; the normalizations can be done in linear time in the input size.</p>
</div>

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
                    {% for j in range(input.ny) %}
                        {% if j < i %}
                            <td class="element dim">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% elif oe.locations and saferatio(safeget(oe.locations,i,j), oe.max_error_limit, 1) %}
                            <td class="element correct">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% elif oe.locations and saferatio(safeget(oe.locations,i,j), oe.max_error_limit, 100) %}
                            <td class="element slightlywrong">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% else %}
                            <td class="element verywrong">{{ safeprint(safeget(output.result,i,j)) }}</td>
                        {% endif %}
                    {% endfor %}
                </tr>
            {% endfor %}
        </table></div></div>
        {% if oe.max_error_limit and oe.locations %}
            <p>Above I have here highlighted the cells that contain wrong values. Here <span class="elementexample slightlywrong">blue shading</span> indicates slightly wrong values (rounding errors perhaps?) while <span class="elementexample verywrong">orange shading</span> indicates values that are very far from what I was expecting (some problems with the code logic?).</p>
        {% endif %}
    {% endif %}
{% endif %}
{% if safenum(oe.max_error) > 0 and safenum(oe.max_error_limit) > 0 %}
    <p>In comparison with the expected output, the largest error was ≈ <strong>{{ safereadable(oe.max_error) }}</strong>.
    In this test the largest errors should be &lt; {{ safereadable(oe.max_error_limit) }}.
    That is, your code made errors that are ≈ {{ safereadable(oe.max_error/oe.max_error_limit) }} times too large.</p>
    {% if saferatio(oe.max_error, oe.max_error_limit, 100) %}
        <p>As the errors were relatively small, could they be maybe rounding errors?</p>
    {% endif %}
{% elif safenum(oe.gvfa_error) and safenum(oe.gvfa_error_limit) %}
    <p>The probabilistic test indicated errors that are ≈ {{ safereadable(oe.gvfa_error/oe.gvfa_error_limit) }} times too large.</p>
{% endif %}
""")
    return Markup(
        templ_basic.render(
            input=test.raw.get("input", {}),
            output=test.raw.get("output", {}),
            oe=test.raw.get("output_errors", {}),
            safeget=safeget,
            safenum=safenum,
            saferatio=saferatio,
            safeprint=safeprint,
            safereadable=safereadable,
        ))


def explain_terminal(r, color=False):
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}

    ny = input.get('ny', None)
    nx = input.get('nx', None)
    data = input.get('data', None)
    result = output.get("result", None)
    locations = oe.get("locations", None)
    max_error = oe.get("max_error", None)
    max_error_limit = oe.get('max_error_limit', None)
    gvfa_error = oe.get("gvfa_error", None)
    gvfa_error_limit = oe.get('gvfa_error_limit', None)

    if color:
        hl, minor, reset = '\033[31;1m', '\033[34;1m', '\033[0m'
        ok1, ok2 = '', ''
        minor1, minor2 = minor, reset
        major1, major2 = hl, reset
    else:
        hl, minor, reset = '', '', ''
        ok1, ok2 = ' ', ' '
        minor1, minor2 = '(', ')'
        major1, major2 = '[', ']'

    expl = ''
    if ny is not None and nx is not None:
        expl += f'In this test I called your function with the following parameters:\n'
        expl += f' · ny = {ny}\n'
        expl += f' · nx = {nx}\n'
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
                for x in range(ny):
                    m1, m2 = '', ''
                    v = safeget(result, y, x)
                    if locations:
                        e = safeget(locations, y, x)
                        if saferatio(e, max_error_limit, 1):
                            m1, m2 = ok1, ok2
                        elif saferatio(e, max_error_limit, 100):
                            m1, m2 = minor1, minor2
                        else:
                            m1, m2 = major1, major2
                    expl += f' {m1}{safeprintt(v)}{m2}'
                expl += '\n'
            expl += '\n'
            if locations is not None and safenum(max_error_limit) > 0:
                expl += f'Above I have highlighted the cells that contain wrong values as follows:\n'
                expl += f' · {minor1}slightly wrong{minor2} — rounding errors perhaps?\n'
                expl += f' · {major1}far from correct{major2} — some problems with the code logic?\n'
                expl += '\n'

    if safenum(max_error) > 0 and safenum(max_error_limit) > 0:
        expl += f'In comparison with the expected output, the largest error was ≈ {hl}{readable(max_error)}{reset}.\n'
        expl += f'In this test the largest errors should be < {readable(max_error_limit)}.\n'
        rel = max_error / max_error_limit
        expl += f'That is, your code made errors that are ≈ {readable(rel)} times too large.\n'
        if rel < 100:
            expl += f'As the errors were relatively small, could they be maybe rounding errors?\n'
        expl += '\n'
    elif safenum(gvfa_error) > 0 and safenum(gvfa_error_limit) > 0:
        rel = gvfa_error / gvfa_error_limit
        expl += f'The probabilistic test indicated errors that are ≈ {readable(rel)} times too large.\n'
        expl += '\n'

    return expl

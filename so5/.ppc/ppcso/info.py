code = "so"
name = "SO"
descr = "sorting"


def isnum(v):
    return v is not None and (isinstance(v, int) or
                              (isinstance(v, float) and math.isfinite(v)))


def safeget(m, i):
    try:
        return m[i]
    except IndexError:
        return None
    except TypeError:
        return None


def safenum(v, default=0):
    if isnum(v):
        return v
    else:
        return default


def safeprint(v, fmt='{:20d}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return str(v)


def safeprintt(v, fmt='{:20d}', sfmt='{:>20s}'):
    if v is None:
        return '–'
    elif isnum(v):
        return fmt.format(v)
    else:
        return sfmt.format(v)


def html():
    from markupsafe import Markup
    return Markup("""
<p>In this task, you will implement parallel sorting algorithms that outperform the single-threaded <code>std::sort</code>.</p>

<h3>Interface</h3>

<p>We have already defined the following data type that represents 64-bit unsigned integers:</p>
<div class="prewrap"><pre>
typedef unsigned long long data_t;
</pre></div>

<p>You need to implement the following function:</p>
<div class="prewrap"><pre>
void psort(int n, data_t* data)
</pre></div>

<p>Here <code>n</code> is the size of the input array <code>data</code>. All input elements are of type <code>data_t</code>, i.e., 64-bit unsigned integers.</p>

<h3>Correct output</h3>

<p>Your function should have exactly the same behavior as:</p>
<div class="prewrap"><pre>
std::sort(data, data+n);
</pre></div>
<p>That is, after calling your function, the array should be in a sorted order. You are free to allocate some additional memory to keep temporary data if needed.</p>


<h3>Rules</h3>

<p>In your implementation, you are permitted and encouraged to use <strong>all single-threaded features of the C++ standard library</strong>. In particular, you can freely use the single-threaded <code>std::sort</code> as a subroutine in your code, as well as all other features of the <a href="http://en.cppreference.com/w/cpp/algorithm">algorithms library</a>.</p>

<p>For multi-threading, you can use the basic primitives of OpenMP, as usual.</p>

<p>Please <strong>do not hard-code the number of threads</strong>; please let OpenMP choose the number of threads, and use e.g. <a href="https://ppc.cs.aalto.fi/ch3/more/">functions omp_get_max_threads and omp_get_num_threads</a> if you need to know the number of threads. This way your code should automatically respect e.g. the <code>OMP_NUM_THREADS</code> environment variable. Your code should work <em>reasonably efficiently</em> for any number of threads between 1 and 20, and it should work <em>correctly</em> also for a larger number of threads. For example, your program must not crash if we change <code>OMP_NUM_THREADS</code> from 20 to 21, and your program should not immediately fall back to a single-threaded implementation  if we change <code>OMP_NUM_THREADS</code> from 20 to 19.</p>

<p>For the GPU exercise, you have to stick to the basic CUDA API. In particular, you are <strong>not allowed to use Thrust</strong>.</p>

<p>In the merge sort task and quicksort task, it is perfectly fine to resort to the single-threaded <code>std::sort</code> in the base case (regardless of what algorithm the standard library happens to use internally); the key requirement is that the multi-threaded parts are based on the idea of merge sort and quicksort, respectively. For merging and partitioning, you are free to use any strategy, and you are encouraged to use also e.g. vector instructions whenever possible.</p>

""")


def explain_web(test):
    from jinja2 import Template
    from markupsafe import Markup

    templ_basic = Template("""
{% if input.n %}
    <p>In this test I called your function with n = {{ input.n }}.</p>
    {% if input.data and output.result and oe.correct %}
        <p>Here is your input, your output, and the expected output:</p>
        <div class="tablewrap"><table>
            <tr>
                <th>Input</th><th>Output</th><th>Expected</th>
            </tr>
            {% for i in range(input.n) %}
                {% if safeget(output.result,i) != safeget(oe.correct,i) %}
                    <tr>
                        <td class="right">{{ safeprint(safeget(input.data,i)) }}</td>
                        <td class="right"><strong>{{ safeprint(safeget(output.result,i)) }}</strong></td>
                        <td class="right"><strong>{{ safeprint(safeget(oe.correct,i)) }}</strong></td>
                    </tr>
                {% else %}
                    <tr>
                        <td class="right">{{ safeprint(safeget(input.data,i)) }}</td>
                        <td class="right">{{ safeprint(safeget(output.result,i)) }}</td>
                        <td class="right">{{ safeprint(safeget(oe.correct,i)) }}</td>
                    </tr>
                {% endif %}
            {% endfor %}
        </table></div>
    {% endif %}
{% endif %}
{% if oe.type == 1 %}
    <p>It seems that the output is not in the correct order.</p>
{% elif oe.type == 2 %}
    <p>It seems that the output contains the wrong set of numbers.</p>
{% endif %}
""")
    return Markup(
        templ_basic.render(
            input=test.raw.get("input", {}),
            output=test.raw.get("output", {}),
            oe=test.raw.get("output_errors", {}),
            safeget=safeget,
            safenum=safenum,
            safeprint=safeprint,
        ))


def explain_terminal(r, color=False):
    input = r.input_data.get('data', None)
    n = r.input_data.get('n', None)
    output = r.output_data.get('result', None)
    error_type = r.output_errors.get('type', None)
    correct = r.output_errors.get('correct', None)

    if color:
        hl, reset = '\033[31;1m', '\033[0m'
    else:
        hl, reset = '', ''

    result = ''
    if n is not None:
        result += f'In this test I called your function with n = {n}.\n\n'

        if input is not None and output is not None and correct is not None:
            result += 'Here is your input, your output, and the expected output:\n\n'
            hinput, houtput, hcorrect = 'INPUT', 'OUTPUT', 'EXPECTED'
            result += f'{hinput:>20s} {houtput:>20s} {hcorrect:>20s}\n'
            for i in range(n):
                x = safeget(input, i)
                y = safeget(output, i)
                z = safeget(correct, i)
                if y != z:
                    result += f'{safeprintt(x)} {hl}{safeprintt(y)} {safeprintt(z)}{reset}\n'
                else:
                    result += f'{safeprintt(x)} {safeprintt(y)} {safeprintt(z)}\n'
            result += '\n'

    if error_type == 1:
        result += 'It seems that the output is not in the correct order.\n'
    elif error_type == 2:
        result += 'It seems that the output contains the wrong set of numbers.\n'

    return result

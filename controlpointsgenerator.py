import numpy as np
import math
import itertools
import multiprocessing
from functools import partial
from scipy.integrate import quad
import cv2
from scipy.interpolate import interp1d
from skimage import measure 



# TODO: faster resampling w/ arclength
# TODO: second derivatives
# TODO: local refinement
# TODO: quadratic prefilters
# TODO: Hermite first order


class SplineCurve:
    wrongDimensionMessage = 'It looks like coefs is a 2D array with second dimension different than two. I don\'t know how to handle this yet.'
    wrongArraySizeMessage = 'It looks like coefs is neither a 1 nor a 2D array. I don\'t know how to handle this yet.'
    noCoefsMessage = 'This model doesn\'t have any coefficients.'

    def __init__(self, M, splineGenerator, closed):
        if M >= splineGenerator.support():
            self.M = M
        else:
            raise RuntimeError('self.M must be greater or equal than the spline generator support size.')
            return

        self.splineGenerator = splineGenerator
        self.halfSupport = self.splineGenerator.support() / 2.0
        self.closed = closed
        self.coefs = None

    def sampleSequential(self, samplingRate):
        if self.coefs is None:
            raise RuntimeError(self.noCoefsMessage)
            return

        if len(self.coefs.shape) == 1 or (len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2):
            if self.closed:
                N = samplingRate * self.M
            else:
                N = (samplingRate * (self.M - 1)) + 1
            curve = [self.parameterToWorld(float(i) / float(samplingRate)) for i in range(0, N)]

        else:
            raise RuntimeError(self.wrongArraySizeMessage)
            return

        return np.stack(curve)

    def sampleParallel(self, samplingRate):
        if self.coefs is None:
            raise RuntimeError(self.noCoefsMessage)
            return

        if len(self.coefs.shape) == 1 or (len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2):
            if self.closed:
                N = samplingRate * self.M
            else:
                N = (samplingRate * (self.M - 1)) + 1

            cpucount = multiprocessing.cpu_count()
            with multiprocessing.Pool(cpucount) as pool:
                iterable = [float(i) / float(samplingRate) for i in range(0, N)]
                res = pool.map(self.parameterToWorld, iterable)

            curve = np.stack(res)
            if len(self.coefs.shape) == 1:
                curve = curve[~np.all(curve == 0)]
            else:
                curve = curve[~np.all(curve == 0, axis=1)]
        else:
            raise RuntimeError(self.wrongArraySizeMessage)
            return

        return curve

#     def getCoefsFromKnots(self, knots):
#         knots = np.array(knots)
#         if len(knots.shape) == 1:
#             if self.closed:
#                 self.coefs = self.splineGenerator.filterPeriodic(knots)
#             else:
#                 self.coefs = self.splineGenerator.filterSymmetric(knots)
#         elif len(knots.shape) == 2:
#             if (knots.shape[1] == 2):
#                 if self.closed:
#                     coefsX = self.splineGenerator.filterPeriodic(knots[:, 0])
#                     coefsY = self.splineGenerator.filterPeriodic(knots[:, 1])
#                 else:
#                     coefsX = self.splineGenerator.filterSymmetric(knots[:, 0])
#                     coefsY = self.splineGenerator.filterSymmetric(knots[:, 1])
#                 self.coefs = np.hstack((np.array([coefsX]).transpose(), np.array([coefsY]).transpose()))
#             else:
#                 raise RuntimeError(self.wrongDimensionMessage)
#                 return
#         else:
#             raise RuntimeError(self.wrongArraySizeMessage)
#             return

#         return self.coefs
    
    def getCoefsFromKnots(self, knots):
        knots = np.array(knots)
        if len(knots.shape) == 1:
            if self.closed:
                self.coefs = self.splineGenerator.filterPeriodic(knots)
            else:
                self.coefs = self.splineGenerator.filterSymmetric(knots)
        elif len(knots.shape) == 4:
            if (knots.shape[3] == 2):
                if self.closed:
                    coefsX = self.splineGenerator.filterPeriodic(knots[:, :, :, 0])
                    coefsY = self.splineGenerator.filterPeriodic(knots[:, :, :, 1])
                else:
                    coefsX = self.splineGenerator.filterSymmetric(knots[:, 0])
                    coefsY = self.splineGenerator.filterSymmetric(knots[:, 1])
                self.coefs = np.stack((coefsX, coefsY), axis = 3)
            else:
                raise RuntimeError(self.wrongDimensionMessage)
                return
        else:
            raise RuntimeError(self.wrongArraySizeMessage)
            return

        return self.coefs

    def getCoefsFromDenseContour(self, contourPoints):
        N = len(contourPoints)
        phi = np.zeros((N, self.M))
        if len(contourPoints.shape) == 1:
            r = np.zeros((N))
        elif len(contourPoints.shape) == 2:
            if (contourPoints.shape[1] == 2):
                r = np.zeros((N, 2))

        if self.closed:
            samplingRate = int(N / self.M)
            extraPoints = N % self.M
        else:
            samplingRate = int(N / (self.M - 1))
            extraPoints = N % (self.M - 1)

        for i in range(0, N):
            r[i] = contourPoints[i]

            if i == 0:
                t = 0
            elif t < extraPoints:
                t += 1.0 / (samplingRate + 1.0)
            else:
                t += 1.0 / samplingRate

            for k in range(0, self.M):
                if self.closed:
                    tval = self.wrapIndex(t, k)
                else:
                    tval = t - k
                if (tval > -self.halfSupport and tval < self.halfSupport):
                    basisFactor = self.splineGenerator.value(tval)
                else:
                    basisFactor = 0.0

                phi[i, k] += basisFactor

        if len(contourPoints.shape) == 1:
            c = np.linalg.lstsq(phi, r, rcond=None)

            self.coefs = np.zeros([self.M])
            for k in range(0, self.M):
                self.coefs[k] = c[0][k]
        elif len(contourPoints.shape) == 2:
            if (contourPoints.shape[1] == 2):
                cX = np.linalg.lstsq(phi, r[:, 0], rcond=None)
                cY = np.linalg.lstsq(phi, r[:, 1], rcond=None)

                self.coefs = np.zeros([self.M, 2])
                for k in range(0, self.M):
                    self.coefs[k] = np.array([cX[0][k], cY[0][k]])

        return self.coefs

    def getCoefsFromBinaryMask(self, binaryMask):
        from skimage import measure
        contours = measure.find_contours(binaryMask, 0)

        if len(contours) > 1:
            raise RuntimeWarning(
                'Multiple objects were found on the binary mask. Only the first one will be processed.')

        coefs = self.getCoefsFromDenseContour(contours[0])
        return coefs
    
    def getCoefsFromBinaryMask_skimage(self, binaryMask):
        contours = self.contour_skimage_mask_uniform(binaryMask)
        coefs = self.getCoefsFromDenseContour(contours)
        return coefs
    
    def getCoefsFromBinaryMask_cv2(self, binaryMask):
        contours = self.contour_cv2_mask_uniform(binaryMask)
        coefs = self.getCoefsFromDenseContour(contours)
        coefs = np.stack((coefs[:,1],coefs[:,0]), axis = -1)
        return coefs
    
    def contour_cv2_mask(self,mask):
        mask = mask.astype(np.uint8)       
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(cnt) for cnt in contours]  
        max_ind = np.argmax(areas)
        contour = np.squeeze(contours[max_ind]).astype(np.float32)
        contour = np.reshape(contour,(-1,2))
        contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)     
        return contour
    
    def contour_cv2_mask_uniform(self,mask):
        num_pts = 400        
        mask = mask.astype(np.uint8)    
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(cnt) for cnt in contours]    
        max_ind = np.argmax(areas)
        contour = np.squeeze(contours[max_ind])
        contour = np.reshape(contour,(-1,2))
        contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)
        contour = contour.astype('float32')
        
        rows,cols = mask.shape
        delta = np.diff(contour,axis=0)
        s = [0]
        for d in delta:
            dl = s[-1] + np.linalg.norm(d)
            s.append(dl)
            
        if (s[-1] == 0):
            s[-1] = 1

        s = np.array(s)/s[-1]
        fx = interp1d(s,contour[:,0]/rows, kind='linear')
        fy = interp1d(s,contour[:,1]/cols, kind='linear')
        S = np.linspace(0,1,num_pts, endpoint = False)
        X = rows * fx(S)
        Y = cols * fy(S)
        
        contour = np.transpose(np.stack([X,Y])).astype(np.float32)
        return contour
    
    def contour_skimage_mask_uniform(self,mask):
        num_pts = 400        
        mask = mask.astype(np.uint8)    
        contours = measure.find_contours(mask, 0)
        contours = contours[0]        
        
        contour = np.reshape(contours,(-1,2))
        contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)
        contour = contour.astype('float32')
        
        rows,cols = mask.shape
        delta = np.diff(contour,axis=0)
        s = [0]
        for d in delta:
            dl = s[-1] + np.linalg.norm(d)
            s.append(dl)
            
        if (s[-1] == 0):
            s[-1] = 1

        s = np.array(s)/s[-1]
        fx = interp1d(s,contour[:,0]/rows, kind='linear')
        fy = interp1d(s,contour[:,1]/cols, kind='linear')
        S = np.linspace(0,1,num_pts, endpoint = False)
        X = rows * fx(S)
        Y = cols * fy(S)
        
        contour = np.transpose(np.stack([X,Y])).astype(np.float32)
        return contour

    def arcLength(self, t0, tf):
        if t0 == tf:
            return 0.0
        elif t0 > tf:
            temp = tf
            tf = t0
            t0 = temp

        integral = quad(lambda t: np.linalg.norm(self.parameterToWorld(t, dt=True)), t0, tf, epsabs=1e-6,
                        epsrel=1e-6, maxp1=50, limit=100)
        return integral[0]


    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(SplineCurve.noCoefsMessage)
            return

        value = 0.0
        for k in range(0, self.M):
            if self.closed:
                tval = self.wrapIndex(t, k)
            else:
                tval = t - k
            if (tval > -self.halfSupport and tval < self.halfSupport):
                if dt:
                    splineValue=self.splineGenerator.firstDerivativeValue(tval)
                else:
                    splineValue=self.splineGenerator.value(tval)
                value += self.coefs[k] * splineValue
        return value

    def wrapIndex(self, t, k):
        wrappedT = t - k
        if k < t - self.halfSupport:
            if k + self.M >= t - self.halfSupport and k + self.M <= t + self.halfSupport:
                wrappedT = t - (k + self.M)
        elif k > t + self.halfSupport:
            if k - self.M >= t - self.halfSupport and k - self.M <= t + self.halfSupport:
                wrappedT = t - (k - self.M)
        return wrappedT

    def centroid(self):
        centroid=np.zeros((2))
        
        for k in range(0, self.M):
            centroid+=self.coefs[k]

        return centroid/self.M

    def translate(self, translationVector):
        for k in range(0, self.M):
            self.coefs[k]+=translationVector

    def scale(self, scalingFactor):
        centroid=self.centroid()
        
        for k in range(0, self.M):
            vectorToCentroid=self.coefs[k]-centroid
            self.coefs[k]=centroid+scalingFactor*vectorToCentroid

    def rotate(self, rotationMatrix):
        for k in range(0, self.M):
            self.coefs[k]=np.matmul(rotationMatrix, self.coefs[k])


class HermiteSplineCurve(SplineCurve):
    coefTangentMismatchMessage = 'It looks like coefs and tangents have different shapes.'

    def __init__(self, M, splineGenerator, closed):
        if not splineGenerator.multigenerator():
            raise RuntimeError('It looks like you are trying to use a single generator to build a multigenerator spline model.')
            return

        SplineCurve.__init__(self, M, splineGenerator, closed)
        self.tangents=None

    def getCoefsFromKnots(self, knots, tangentAtKnots):
        knots = np.array(knots)
        tangentAtKnots = np.array(tangentAtKnots)

        if knots.shape != tangentAtKnots.shape:
            raise RuntimeError(coefTangentMismatchMessage)
            return

        if len(knots.shape) == 1:
            self.coefs = knots
            self.tangents = tangentAtKnots
        elif len(knots.shape) == 2:
            if knots.shape[1] == 2:
                self.coefs = knots
                self.tangents = tangentAtKnots
            else:
                raise RuntimeError(SplineCurve.wrongDimensionMessage)
                return
        else:
            raise RuntimeError(SplineCurve.wrongArraySizeMessage)
            return

        return

    def getCoefsFromDenseContour(self, contourPoints, tangentAtPoints):
        # TODO
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return 

    def getCoefsFromBinaryMask(self, binaryMask):
        # TODO
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return 

    def arcLength(self, t0, tf):
        # TODO
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return 

    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(noCoefsMessage)
            return

        value = 0.0
        for k in range(0, self.M):
            if self.closed:
                tval = self.wrapIndex(t, k)
            else:
                tval = t - k
            if (tval > -self.halfSupport and tval < self.halfSupport):
                if dt:
                    splineValue=self.splineGenerator.firstDerivativeValue(tval)
                else:
                    splineValue=self.splineGenerator.value(tval)
                value += self.coefs[k] * splineValue[0] + self.tangents[k] * splineValue[1]
        return value

    def scale(self, scalingFactor):
        SplineCurve.scale(self, scalingFactor)

        for k in range(0, self.M):
            self.tangents[k]*=scalingFactor

    def rotate(self, rotationMatrix):
        SplineCurve.rotate(self, rotationMatrix)

        for k in range(0, self.M):
            self.tangents[k]=np.matmul(rotationMatrix, self.tangents[k])


class SplineGenerator:
    unimplementedMessage = 'This function is not implemented.'

    def multigenerator(self):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def value(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def firstDerivativeValue(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def secondDerivativeValue(self, x):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def filterSymmetric(self, s):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def filterPeriodic(self, s):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def refinementMask(self):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return

    def support(self):
        # This needs to be overloaded
        raise NotImplementedError(SplineGenerator.unimplementedMessage)
        return


class B1(SplineGenerator):
    def multigenerator(self):
        return False

    def value(self, x):
        val = 0.0
        if 0 <= abs(x) and abs(x) < 1:
            val = 1.0 - abs(x)
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if -1.0 <= x and x < 0:
            val = -1.0
        elif 0 <= x and x <= 1:
            val = 1.0
        return val

    def filterSymmetric(self, s):
        return s

    def filterPeriodic(self, s):
        return s

    def refinementMask(self):
        order = int(self.support())
        mask = np.zeros((order + 1))
        multinomial(order, 2, np.zeros((2)), 0, 2, order, mask)
        return mask

    def support(self):
        return 2.0


class B2(SplineGenerator):
    def multigenerator(self):
        return False

    def value(self, x):
        val = 0.0
        if -1.5 <= x and x <= -0.5:
            val = 0.5 * (x ** 2) + 1.5 * x + 1.125
        elif -0.5 < x and x <= 0.5:
            val = -x * x + 0.75
        elif 0.5 < x and x <= 1.5:
            val = 0.5 * (x ** 2) - 1.5 * x + 1.125
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if -1.5 <= x and x <= -0.5:
            val = x + 1.5
        elif -0.5 < x and x <= 0.5:
            val = -2.0 * x
        elif 0.5 < x and x <= 1.5:
            val = x - 1.5
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if -1.5 <= x and x <= -0.5:
            val = 1.0
        elif -0.5 < x and x <= 0.5:
            val = -2.0
        elif 0.5 < x and x <= 1.5:
            val = 1.0
        return val

    def refinementMask(self):
        order = int(self.support())
        mask = np.zeros((order + 1))
        multinomial(order, 2, np.zeros((2)), 0, 2, order, mask)
        return mask

    def support(self):
        return 3.0


class B3(SplineGenerator):
    def multigenerator(self):
        return False

    def value(self, x):
        val = 0.0
        if 0 <= abs(x) and abs(x) < 1:
            val = 2.0 / 3.0 - (abs(x) ** 2) + (abs(x) ** 3) / 2.0
        elif 1 <= abs(x) and abs(x) <= 2:
            val = ((2.0 - abs(x)) ** 3) / 6.0
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x < 1:
            val = -2.0 * x + 1.5 * x * x
        elif -1 < x and x < 0:
            val = -2.0 * x - 1.5 * x * x
        elif 1 <= x and x <= 2:
            val = -0.5 * ((2.0 - x) ** 2)
        elif -2 <= x and x <= -1:
            val = 0.5 * ((2.0 + x) ** 2)
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x < 1:
            val = -2.0 + 3.0 * x
        elif -1 < x and x < 0:
            val = -2.0 - 3.0 * x
        elif 1 <= x and x <= 2:
            val = (2.0 - x)
        elif -2 <= x and x <= -1:
            val = (2.0 + x)
        return val

    def filterSymmetric(self, s):
        self.M = len(s)
        pole = -2.0 + math.sqrt(3.0)

        cp = np.zeros(self.M)
        eps = 1e-8
        k0 = np.min(((2 * self.M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole))))))
        for k in range(0, k0):
            k = k % (2 * self.M - 2)
            if k >= self.M:
                val = s[2 * self.M - 2 - k]
            else:
                val = s[k]
            cp[0] += (val * (pole ** k))
        cp[0] *= (1.0 / (1.0 - (pole ** (2 * self.M - 2))))

        for k in range(1, self.M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(self.M)
        cm[self.M - 1] = cp[self.M - 1] + (pole * cp[self.M - 2])
        cm[self.M - 1] *= (pole / ((pole ** 2) - 1))
        for k in range(self.M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6.0

        c[np.where(abs(c) < eps)] = 0.0
        return c

#     def filterPeriodic(self, s):
#         self.M = len(s)
#         pole = -2.0 + math.sqrt(3.0)

#         cp = np.zeros(self.M)
#         for k in range(0, self.M):
#             cp[0] += (s[(self.M - k) % self.M] * (pole ** k))
#         cp[0] *= (1.0 / (1.0 - (pole ** self.M)))

#         for k in range(1, self.M):
#             cp[k] = s[k] + pole * cp[k - 1]

#         cm = np.zeros(self.M)
#         for k in range(0, self.M):
#             cm[self.M - 1] += ((pole ** k) * cp[k])
#         cm[self.M - 1] *= (pole / (1.0 - (pole ** self.M)))
#         cm[self.M - 1] += cp[self.M - 1]
#         cm[self.M - 1] *= (-pole)

#         for k in range(self.M - 2, -1, -1):
#             cm[k] = pole * (cm[k + 1] - cp[k])

#         c = cm * 6.0

#         eps = 1e-8
#         c[np.where(abs(c) < eps)] = 0.0
#         return c

    def filterPeriodic(self, s):
        self.M = s.shape[2]
        pole = -2.0 + math.sqrt(3.0)

        cp = np.zeros_like(s).astype('float')
        for k in range(0, self.M):
            cp[:,:,0] += (s[:,:,(self.M - k) % self.M] * (pole ** k))
        cp[:,:,0] *= (1.0 / (1.0 - (pole ** self.M)))

        for k in range(1, self.M):
            cp[:,:,k] = s[:,:,k] + pole * cp[:,:,k - 1]

        cm = np.zeros_like(s).astype('float')
        for k in range(0, self.M):
            cm[:,:,self.M - 1] += ((pole ** k) * cp[:,:,k])
        cm[:,:,self.M - 1] *= (pole / (1.0 - (pole ** self.M)))
        cm[:,:,self.M - 1] += cp[:,:,self.M - 1]
        cm[:,:,self.M - 1] *= (-pole)

        for k in range(self.M - 2, -1, -1):
            cm[:,:,k] = pole * (cm[:,:,k + 1] - cp[:,:,k])

        c = cm * 6.0

        eps = 1e-8
        c[np.where(abs(c) < eps)] = 0.0
        return c

    def refinementMask(self):
        order = int(self.support())
        mask = np.zeros((order + 1))
        multinomial(order, 2, np.zeros((2)), 0, 2, order, mask)
        return mask

    def support(self):
        return 4.0


class EM(SplineGenerator):
    def multigenerator(self):
        return False

    def __init__(self, M, alpha):
        self.M = M
        self.alpha = alpha

    def value(self, x):
        x += (self.support() / 2.0);
        L = (math.sin(math.pi / self.M) / (math.pi / self.M)) ** (-2)

        val = 0.0
        if x >= 0 and x < 1:
            val = 2.0 * math.sin(self.alpha * 0.5 * x) * math.sin(self.alpha * 0.5 * x)
        elif x >= 1 and x < 2:
            val = (math.cos(self.alpha * (x - 2)) + math.cos(self.alpha * (x - 1)) - 2.0 * math.cos(self.alpha))
        elif x >= 2 and x <= 3:
            val = 2.0 * math.sin(self.alpha * 0.5 * (x - 3)) * math.sin(self.alpha * 0.5 * (x - 3))

        return (L * val) / (self.alpha * self.alpha)

    def firstDerivativeValue(self, x):
        x += (self.support() / 2.0);
        L = (math.sin(math.pi / self.M) / (math.pi / self.M)) ** (-2)

        val = 0.0
        if 0 <= x and x <= 1:
            val = self.alpha * math.sin(self.alpha * x)
        elif 1 < x and x <= 2:
            val = self.alpha * (math.sin(self.alpha * (1 - x)) + math.sin(self.alpha * (2 - x)))
        elif 2 < x and x <= 3:
            val = self.alpha * math.sin(self.alpha * (x - 3))

        return (L * val) / (self.alpha * self.alpha)

    def secondDerivativeValue(self, x):
        x += (self.support() / 2.0);
        L = (math.sin(math.pi / self.M) / (math.pi / self.M)) ** (-2)

        val = 0.0
        if 0 <= x and x <= 1:
            val = self.alpha * self.alpha * math.cos(self.alpha * x)
        elif 1 < x and x <= 2:
            val = self.alpha * self.alpha * (-math.cos(self.alpha * (1 - x)) - math.cos(self.alpha * (2 - x)))
        elif 2 < x and x <= 3:
            val = self.alpha * self.alpha * self.cos(self.alpha * (x - 3))

        return (L * val) / (self.alpha * self.alpha)

    def filterSymmetric(self, s):
        self.M = len(s)
        b0 = self.value(0)
        b1 = self.value(1)
        pole = (-b0 + math.sqrt(2.0 * b0 - 1.0)) / (1.0 - b0)

        cp = np.zeros(self.M)
        eps = 1e-8
        k0 = np.min(((2 * self.M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole))))))
        for k in range(0, k0):
            k = k % (2 * self.M - 2)
            if k >= self.M:
                val = s[2 * self.M - 2 - k]
            else:
                val = s[k]
            cp[0] += (val * (pole ** k))
        cp[0] *= (1.0 / (1.0 - (pole ** (2 * self.M - 2))))

        for k in range(1, self.M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(self.M)
        cm[self.M - 1] = cp[self.M - 1] + (pole * cp[self.M - 2])
        cm[self.M - 1] *= (pole / ((pole * pole) - 1))
        for k in range(self.M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm / b1

        c[np.where(abs(c) < eps)] = 0.0
        return c

    def filterPeriodic(self, s):
        self.M = len(s)
        b0 = self.value(0)
        pole = (-b0 + math.sqrt(2.0 * b0 - 1.0)) / (1.0 - b0)

        cp = np.zeros(self.M)
        cp[0] = s[0]
        for k in range(0, self.M):
            cp[0] += (s[k] * (pole ** (self.M - k)))
        cp[0] *= (1.0 / (1.0 - (pole ** self.M)))

        for k in range(1, self.M):
            cp[k] = s[k] + (pole * cp[k - 1])

        cm = np.zeros(self.M)
        cm[self.M - 1] = cp[self.M - 1]
        for k in range(0, self.M - 1):
            cm[self.M - 1] += (cp[k] * (pole ** (k + 1)))
        cm[self.M - 1] *= (1.0 / (1.0 - (pole ** self.M)))
        cm[self.M - 1] *= ((1 - pole) ** 2)

        for k in range(self.M - 2, -1, -1):
            cm[k] = (pole * cm[k + 1]) + (((1 - pole) ** 2) * cp[k])

        c = cm

        eps = 1e-8
        c[np.where(abs(c) < eps)] = 0.0
        return c

    def refinementMask(self):
        order = int(self.support())
        mask = np.zeros((order + 1))

        denominator = 2.0 ** (order - 1.0)
        mask[0] = 1.0 / denominator
        mask[1] = (2.0 * math.cos(self.alpha) + 1.0) / denominator
        mask[2] = mask[1]
        mask[3] = 1.0 / denominator

        return mask

    def support(self):
        return 3.0


class Keys(SplineGenerator):
    def multigenerator(self):
        return False

    def value(self, x):
        val = 0.0
        if 0 <= abs(x) and abs(x) <= 1:
            val = (3.0 / 2.0) * (abs(x) ** 3) - (5.0 / 2.0) * (abs(x) ** 2) + 1
        elif 1 < abs(x) and abs(x) <= 2:
            val = (-1.0 / 2.0) * (abs(x) ** 3) + (5.0 / 2.0) * (abs(x) ** 2) - 4.0 * abs(x) + 2.0
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x <= 1:
            val = x * (4.5 * x - 5.0)
        elif -1 <= x and x < 0:
            val = -x * (4.5 * x + 5.0)
        elif 1 < x and x <= 2:
            val = -1.5 * x * x + 5.0 * x - 4.0
        elif -2 <= x and x < -1:
            val = 1.5 * x * x + 5.0 * x + 4.0
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x <= 1:
            val = 9.0 * x - 5.0
        elif -1 <= x and x < 0:
            val = -9.0 * x - 5.0
        elif 1 < x and x <= 2:
            val = -3.0 * x + 5.0
        elif -2 <= x and x < -1:
            val = 3.0 * x + 5.0
        return val

    def filterSymmetric(self, s):
        return s

    def filterPeriodic(self, s):
        return s

    def support(self):
        return 4.0


class H3(SplineGenerator):
    def multigenerator(self):
        return True

    def value(self, x):
        return np.array([self.h31(x), self.h32(x)])

    def h31(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = (1.0 + (2.0 * x)) * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = (1.0 - (2.0 * x)) * (x + 1) * (x + 1)
        return val

    def h32(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = x * (x - 1) * (x - 1)
        elif x < 0 and x >= -1:
            val = x * (x + 1) * (x + 1)
        return val

    def firstDerivativeValue(self, x):
        return np.array([self.h31prime(x), self.h32prime(x)])

    def h31prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = 6.0 * x * (x - 1.0)
        elif x < 0 and x >= -1:
            val = -6.0 * x * (x + 1.0)
        return val

    def h32prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            val = 3.0 * x * x - 4.0 * x + 1
        elif x < 0 and x >= -1:
            val = 3.0 * x * x + 4.0 * x + 1
        return val

    def support(self):
        return 2.0

class HE3(SplineGenerator):
    def multigenerator(self):
        return True

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, x):
        return np.array([self.he31(x), self.he32(x)])

    def he31(self, x):
        val = 0.0
        if x >= 0:
            val = self.g1(x)
        else:
            val = self.g1(-x)
        return val

    def g1(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (0.5 * self.alpha * math.cos(0.5 * self.alpha)) - math.sin(0.5 * self.alpha)
            num = (0.5 * ((self.alpha * math.cos(0.5 * self.alpha)) - math.sin(0.5 * self.alpha))) - (0.5 * self.alpha * math.cos(0.5 * self.alpha) * x) - (0.5 * math.sin(0.5 * self.alpha - (self.alpha * x)))
            val = num / denom
        return val

    def he32(self, x):
        val = 0.0
        if x >= 0:
            val = self.g2(x)
        else:
            val = -1.0 * self.g2(-x)
        return val

    def g2(self, x):
        val = 0.0;
        if x >= 0 and x <= 1:
            denom = ((0.5 * self.alpha * math.cos(0.5 * self.alpha)) - math.sin(0.5 * self.alpha)) * (4.0 * self.alpha) * math.sin(0.5 * self.alpha);
            num = -((self.alpha * math.cos(self.alpha)) - math.sin(self.alpha)) - (2.0 * self.alpha * math.sin(0.5 * self.alpha) * math.sin(0.5 * self.alpha) * x) - (2.0 * math.sin(0.5 * self.alpha) * math.cos(self.alpha * (x - 0.5))) + (self.alpha * math.cos(self.alpha * (x - 1)))
            val = num / denom
        return val

    def firstDerivativeValue(self, x):
        return np.array([self.he31prime(x), self.he32prime(x)])

    def he31prime(self, x):
        val = 0.0
        if x >= 0:
            val = self.g1pPrime(x)
        else:
            val =  -1.0 * self.g1prime(-x)
        return val

    def g1prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = (0.5 * self.alpha * math.cos(0.5 * self.alpha)) - math.sin(0.5 * self.alpha)
            num = -(0.5 * self.alpha * math.cos(0.5 * self.alpha)) + (0.5 * self.alpha * math.cos(0.5 * self.alpha - (self.alpha * x)))
            val = num / denom
        return val

    def he32prime(self, x):
        val = 0.0
        if x >= 0:
            val = self.g2prime(x)
        else:
            val = self.g2prime(-x)
        return val

    def g2prime(self, x):
        val = 0.0
        if x >= 0 and x <= 1:
            denom = ((0.5 * self.alpha * math.cos(0.5 * self.alpha)) - math.sin(0.5 * self.alpha)) * (4.0 * self.alpha) * math.sin(0.5 * self.alpha)
            num = -(2.0 * self.alpha * math.sin(0.5 * self.alpha) * math.sin(0.5 * self.alpha)) + (2.0 * self.alpha * math.sin(0.5 * self.alpha) * math.sin(self.alpha * (x - 0.5))) - (self.alpha * self.alpha * math.sin(self.alpha * (x - 1)))
            val = num / denom
        return val

    def support(self):
        return 2.0


# Recursive function to compute the multinomial coefficient of (x0+x1+...+xm-1)^N
# This function finds every {k0,...,km-1} such that k0+...+km-1=N
# (cf multinomial theorem on Wikipedia for a detailed explanation)
def multinomial(maxValue, numberOfCoefficiens, kArray, iteration, dilationFactor, order, mask):
    if numberOfCoefficiens == 1:
        kArray[iteration] = maxValue

        denominator = 1.0
        degree = 0
        for k in range(0, dilationFactor):
            denominator *= np.math.factorial(kArray[k])
            degree += (k * kArray[k])

        coef = np.math.factorial(order) / denominator
        mask[int(degree)] += (coef / (dilationFactor ** (order - 1)))

    else:
        for k in range(0, maxValue + 1):
            kArray[iteration] = k
            multinomial(maxValue - k, numberOfCoefficiens - 1, kArray, iteration + 1, dilationFactor, order, mask)

    return
